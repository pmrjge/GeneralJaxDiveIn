import functools as ft
import pickle
import os

import jax
import jax.nn as jnn
import jax.random as jr
import jax.numpy as jnp

from jax.scipy.special import logsumexp

import numpy as np

import optax

import einops

import haiku as hk
import haiku.initializers as hki
import pandas as pd

import tensorflow as tf
from tqdm import tqdm

from jax.experimental import PartitionSpec
from jax.experimental.pjit import pjit
from jax.experimental.maps import Mesh

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    print(visible_devices)
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    print("Cannot change virtual devices")


class PreProcessPatches:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, images):
        with tf.device('/CPU:0'):
            batch_size = images.shape[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID"
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return jnp.array(patches.numpy(), dtype=jnp.float32)


class PatchEncoder(hk.Module):
    def __init__(self, num_patches, projection_dim=1024):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.positions = jnp.arange(0, self.num_patches, step=1)

    def __call__(self, patch):
        w_init = hki.VarianceScaling()
        b_init = hki.Constant(0)
        return hk.Linear(output_size=self.projection_dim, w_init=w_init, b_init=b_init, name="projection")(patch) + \
               hk.Embed(vocab_size=self.num_patches, embed_dim=self.projection_dim, w_init=w_init,
                        name="position_embed")(self.positions)


class MLP(hk.Module):
    def __init__(self, hidden_units, dropout):
        super(MLP, self).__init__()
        self.hidden_units = hidden_units
        self.dropout = dropout

    def __call__(self, x, *, is_training: bool):
        dropout = self.dropout if is_training else 0.0
        w_init = hki.VarianceScaling()
        b_init = hki.Constant(0)
        for units in self.hidden_units:
            x = hk.Linear(units, w_init=w_init, b_init=b_init)(x)
            x = jnn.gelu(x, approximate=False)
            x = hk.dropout(hk.next_rng_key(), dropout, x)
        return x


class ViT(hk.Module):
    def __init__(self, num_patches=12 * 12, projection_dim=1024, num_blocks=8, num_heads=8, transformer_units_1=2048,
                 transformer_units_2=1024, mlp_head_units=(2048, 1024), dropout=0.5):
        super(ViT, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.transformer_units_1 = transformer_units_1
        self.transformer_units_2 = transformer_units_2
        self.mlp_head_units = mlp_head_units
        self.dropout = dropout
        self.norm = lambda: hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=1e-6,
                                         scale_init=hki.Constant(1.0), offset_init=hki.Constant(0.0))

    def __call__(self, patches, *, is_training: bool):
        dropout = self.dropout if is_training else 0.0

        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)

        w_init = hki.VarianceScaling()
        for _ in range(self.num_blocks):
            x1 = self.norm()(encoded_patches)
            attention = hk.MultiHeadAttention(self.num_heads, self.projection_dim // self.num_heads, w_init=w_init)(x1,
                                                                                                                    x1,
                                                                                                                    x1)
            x2 = attention + encoded_patches
            x3 = self.norm()(x2)
            x3 = MLP((self.transformer_units_1, self.transformer_units_2), self.dropout)(x3, is_training=is_training)

            encoded_patches = x3 + x2

        representation = self.norm()(encoded_patches)
        representation = einops.rearrange(representation, 'b h t -> b (h t)')
        representation = hk.dropout(hk.next_rng_key(), dropout, representation)

        features = MLP(self.mlp_head_units, self.dropout)(representation, is_training=is_training)

        logits = hk.Linear(10)(features)

        return logits - logsumexp(logits, axis=1, keepdims=True)


# Load dataset
with open('../data/digits/data2.dict', 'rb') as f:
    data = pickle.load(f)

x = data['train']
y = data['labels']
xt = data['test']


def process_epoch_gen(a, b, batch_size, patch_size, num_devices):
    proc = PreProcessPatches(patch_size=patch_size)

    topo = batch_size // num_devices

    def epoch_generator(rng):
        n = a.shape[0]
        num_batches = n // batch_size
        key, rng = jr.split(rng)

        perm = jr.permutation(key, n)
        for i in range(num_batches):
            i0 = i * batch_size
            i1 = (i + 1) * batch_size
            subp = perm[i0: i1]
            outx = jnp.array(proc(a[subp]), dtype=jnp.float32)
            outy = jnp.array(b[subp], dtype=jnp.int32)
            yield outx.reshape(num_devices, topo, *outx.shape[1:]), outy.reshape(num_devices, topo, *outy.shape[1:])

    return epoch_generator


batch_size = 2
patch_size = 12

process_gen = process_epoch_gen(x, y, batch_size, patch_size, jax.device_count())

patch_dim = 72 // patch_size


# def build_forward_fn(num_patches=patch_dim * patch_dim, projection_dim=1024, num_blocks=64, num_heads=8, transformer_units_1=2048, transformer_units_2=1024, mlp_head_units=(2048, 1024), dropout=0.5):
def build_forward_fn(num_patches=patch_dim * patch_dim, projection_dim=1024, num_blocks=101, num_heads=8,
                     transformer_units_1=2048, transformer_units_2=1024, mlp_head_units=(512, 256), dropout=0.4):
    def forward_fn(dgt: jnp.ndarray, *, is_training: bool) -> jnp.ndarray:
        return ViT(num_patches=num_patches, projection_dim=projection_dim,
                   num_blocks=num_blocks, num_heads=num_heads, transformer_units_1=transformer_units_1,
                   transformer_units_2=transformer_units_2, mlp_head_units=mlp_head_units,
                   dropout=dropout)(dgt, is_training=is_training)

    return forward_fn


ffn = build_forward_fn()

ffn = hk.transform_with_state(ffn)

apply = ffn.apply

l_apply = ft.partial(apply, is_training=True)
l_apply = jax.jit(l_apply)
fast_apply = jax.jit(apply, static_argnames=('is_training',))

rng = jr.PRNGKey(101)


def focal_loss(labels, y_pred, ce, gamma, alpha):
    weight = labels * jnp.power(1 - y_pred, gamma)
    f_loss = alpha * (weight * ce)
    f_loss = jnp.sum(f_loss, axis=1)
    f_loss = jnp.mean(f_loss, axis=0)
    return f_loss


# @ft.partial(jax.jit, static_argnums=(0, 6))
def ce_loss_fn(forward_fn, params, state, rng, a, b, num_classes: int = 10):
    with Mesh(np.array(jax.devices()), ('devices',)):
        logits, state = forward_fn(params, state, rng, a)

    labels = jnn.one_hot(b, num_classes=num_classes)

    # Weight decay
    l2_loss = 0.5 * jnp.sum(jnp.array([jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)], dtype=jnp.float32))
    l1_loss = jnp.sum(jnp.array([jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params)], dtype=jnp.float32))

    # Normalized CE loss and Focal loss so it is more smooth and gives back better feedback
    # logits = jnp.clip(logits, a_min=jnp.log(1e-12), a_max=jnp.log(1 - 1e-12))
    ce = -labels * logits

    y_pred = jnp.exp(logits)
    # CE loss
    #ce_loss = jnp.sum(ce, axis=1)
    #ce_loss = jnp.mean(ce_loss, axis=0)

    # Focal Loss
    f_loss = focal_loss(labels, y_pred, ce, 2.0, 4.0) + focal_loss(labels, y_pred, ce, 3.0, 4.0) # + focal_loss(labels, y_pred, ce, 4.0, 4.0)

    # Double Soft F1 Loss
    # tp = jnp.sum(labels * y_pred, axis=0)
    # fp = jnp.sum((1 - labels) * y_pred, axis=0)
    # fn = jnp.sum(labels * (1 - y_pred), axis=0)
    # tn = jnp.sum((1 - labels) * (1 - y_pred), axis=0)
    # soft_f11 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    # soft_f10 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    # cost1 = 1 - soft_f11
    # cost0 = 1 - soft_f10
    # f1_cost = jnp.mean(0.5 * (cost1 + cost0))

    # soft f1 score loss + focal loss and weight decay and l1 loss
    return f_loss + 1e-13 * (l2_loss + l1_loss), state


loss_fn = ft.partial(ce_loss_fn, l_apply)

learning_rate = 1e-4
grad_clip_value = 1.0
# scheduler = optax.exponential_decay(init_value=learning_rate, transition_steps=6000, decay_rate=0.99)

optimizer = optax.chain(
    optax.adaptive_grad_clip(grad_clip_value),
    # optax.sgd(learning_rate=learning_rate, momentum=0.99, nesterov=True),
    # optax.scale_by_radam(b1=0.9, eps=1e-8),
    optax.scale_by_adam(b1=0.9, eps=1e-4),
    # optax.scale_by_yogi(),
    # optax.scale_by_schedule(scheduler),
    optax.scale(-learning_rate)
)


class ParamsUpdater:
    def __init__(self, net_init, loss, optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss = loss
        self._opt = optimizer

    def init(self, main_rng, x):
        out_rng, init_rng = jax.random.split(main_rng)
        params, state = self._net_init(init_rng, x, is_training=False)
        opt_state = self._opt.init(params)
        return jnp.array(0), out_rng, params, state, opt_state

    def update(self, num_steps, rng, params, state, opt_state, bx: jnp.ndarray, by: jnp.ndarray):
        rng, new_rng = jax.random.split(rng)

        (loss, state), grads = jax.value_and_grad(self._loss, has_aux=True)(params, state, rng, bx, by)

        grads = jax.lax.psum(grads, axis_name='devices')

        updates, opt_state = self._opt.update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)

        metrics = {
            'step': num_steps,
            'loss': loss,
        }

        return num_steps + 1, new_rng, params, state, opt_state, metrics


updater = ParamsUpdater(ffn.init, loss_fn, optimizer)

print("Initializing parameters..........................")
rng1, rng2 = jr.split(rng)

epoch_gen_temp = process_gen(rng1)
bx, _ = next(epoch_gen_temp)
b = jnp.expand_dims(bx[0, 0, :, :], axis=0)
num_steps, rng, params, state, opt_state = updater.init(rng2, b)

# Training loop
print("Starting training loop..........................")
num_epochs = 2

upd_fn = updater.update

batch_update = jax.pmap(upd_fn, axis_name='devices', in_axes=0, out_axes=0)

params = jax.device_put_replicated(params, devices=jax.devices())
state = jax.device_put_replicated(state, devices=jax.devices())
opt_state = jax.device_put_replicated(opt_state, devices=jax.devices())
num_steps = jax.device_put_replicated(num_steps, devices=jax.devices())


def replicate_tree(t, num_devices):
    return jax.tree_util.tree_map(lambda x: jnp.array([x] * num_devices), t)


n_devices = len(jax.local_devices())

for i in range(num_epochs):
    rng1, rng2, rng = jr.split(rng, 3)
    rng2 = jax.device_put_replicated(rng2, devices=jax.local_devices())
    for step, (bx, by) in tqdm(enumerate(process_gen(rng1)), total=42000 // batch_size):

        bbx = []
        bby = []
        for k in range(n_devices):
            bbx.append(bx[k])
            bby.append(by[k])

        dbx = jax.device_put_sharded(bbx, devices=jax.local_devices())
        dby = jax.device_put_sharded(bby, devices=jax.local_devices())

        num_steps, rng2, params, state, opt_state, metrics = batch_update(num_steps, rng2, params, state, opt_state, dbx, dby)
        if (step + 1) % 8 == 0:
            print(f"......Epoch {i} | Step {step} | Metrics\n\n{metrics} .....................................")

print("Starting evaluation loop........................")
params = jax.device_get(jax.tree_util.tree_map(lambda g: g[0], params))
state = jax.device_get(jax.tree_util.tree_map(lambda g: g[0], state))

res = np.zeros(xt.shape[0], dtype=np.int64)

bts = 10
count = xt.shape[0] // bts

proc = PreProcessPatches(patch_size=patch_size)

for j in tqdm(range(count)):
    rng, = jr.split(rng, 1)
    a, b = j * bts, (j + 1) * bts
    pbt = proc(xt[a:b, :, :, :])
    logits, _ = fast_apply(params, state, rng, pbt, is_training=False)
    res[a:b] = np.array(jnp.argmax(jnp.exp(logits), axis=1), dtype=np.int64)

df = pd.DataFrame({'ImageId': np.arange(1, xt.shape[0] + 1, dtype=np.int64), 'Label': res})

df.to_csv('../data/digits/results.csv', index=False)
