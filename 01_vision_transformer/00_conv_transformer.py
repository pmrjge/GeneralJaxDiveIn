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

from tqdm.auto import tqdm


def img_to_patch(x, patch_size):
    b, h, w, c = x.shape

    x = np.reshape(x, (b, h // patch_size, patch_size, w // patch_size, patch_size, c))
    x = np.transpose(x, (0, 1, 3, 2, 4, 5))
    x = np.reshape(x, (b, -1, *x.shape[3:]))
    return x

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


class TimeDistributed(hk.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def __call__(self, x):

        if len(x.shape) <= 2:
            return self.module(x)

        x_reshape = einops.rearrange(x, 'b t h w c -> (b t) h w c')
        y = self.module(x_reshape)

        return einops.rearrange(y, '(b t) h w c -> b t h w c', b=x.shape[0], t=x.shape[1])


class ConvolutionalBase(hk.Module):

    def __init__(self, dropout):
        self.dropout = dropout
        super().__init__()

    def __call__(self, inputs, is_training=False):
        dropout = self.dropout if is_training else 0.0
        lc_init = hki.VarianceScaling(1.0, 'fan_in', 'truncated_normal')

        x = TimeDistributed(hk.Conv2D(output_channels=32, kernel_shape=3, stride=1, padding="SAME", w_init=lc_init,
                               b_init=hki.RandomNormal(stddev=1e-6)))(inputs)
        x = jnn.gelu(x, approximate=False)
        x = TimeDistributed(hk.MaxPool(window_shape=2, strides=2, padding="SAME"))(x)

        x = TimeDistributed(hk.Conv2D(output_channels=64, kernel_shape=3, stride=1, padding="SAME", w_init=lc_init,
                                      b_init=hki.RandomNormal(stddev=1e-6)))(x)
        x = jnn.gelu(x, approximate=False)
        x = TimeDistributed(hk.MaxPool(window_shape=2, strides=2, padding="SAME"))(x)

        x = TimeDistributed(hk.Conv2D(output_channels=64, kernel_shape=3, stride=1, padding="SAME", w_init=lc_init,
                                      b_init=hki.RandomNormal(stddev=1e-6)))(x)
        x = jnn.gelu(x, approximate=False)
        x = TimeDistributed(hk.MaxPool(window_shape=2, strides=2, padding="SAME"))(x)

        return einops.rearrange(x, 'b c h t f -> b c (h t f)')




class ViT(hk.Module):
    def __init__(self, num_patches=12 * 12, patch_size=8, projection_dim=1024, num_blocks=8, num_heads=8, transformer_units_1=2048,
                 transformer_units_2=1024, mlp_head_units=(2048, 1024), dropout=0.5):
        super(ViT, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.transformer_units_1 = transformer_units_1
        self.transformer_units_2 = transformer_units_2
        self.mlp_head_units = mlp_head_units
        self.dropout = dropout
        self.norm = lambda: hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=1e-6,
                                         scale_init=hki.Constant(1.0), offset_init=hki.Constant(0.0))

    def __call__(self, x, *, is_training: bool):
        dropout = self.dropout if is_training else 0.0

        patches = img_to_patch(x, self.patch_size)

        b, t, _, _, _ = patches.shape

        patches = ConvolutionalBase(self.dropout)(patches, is_training=is_training)

        patches = hk.Linear(self.projection_dim)(patches)
        cls_token = hk.get_parameter('cls_token', (1, 1, self.projection_dim), init=hki.RandomNormal(stddev=1.0)).repeat(b, axis=0)
        encoded_patches = jnp.concatenate([cls_token, patches], axis=1)
        encoded_patches = encoded_patches + hk.get_parameter('pos_embedding', (1, 1 + self.num_patches, self.projection_dim), init=hki.RandomNormal(stddev=1.0))[:, :(t+1)]

        encoded_patches = hk.dropout(hk.next_rng_key(), dropout, encoded_patches)
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
            outx = jnp.array(a[subp], dtype=jnp.float32)
            outy = jnp.array(b[subp], dtype=jnp.int32)
            yield outx.reshape(num_devices, topo, *outx.shape[1:]), outy.reshape(num_devices, topo, *outy.shape[1:])

    return epoch_generator


batch_size = 6
patch_size = 8

process_gen = process_epoch_gen(x, y, batch_size, patch_size, jax.local_device_count())

patch_dim = 80 // patch_size


def build_forward_fn(num_patches=patch_dim * patch_dim, patch_size=patch_size, projection_dim=512, num_blocks=16, num_heads=16,
                     transformer_units_1=2048, transformer_units_2=512, mlp_head_units=(2048, 512), dropout=0.4):
    def forward_fn(dgt: jnp.ndarray, *, is_training: bool) -> jnp.ndarray:
        return ViT(num_patches=num_patches, patch_size=patch_size, projection_dim=projection_dim,
                   num_blocks=num_blocks, num_heads=num_heads, transformer_units_1=transformer_units_1,
                   transformer_units_2=transformer_units_2, mlp_head_units=mlp_head_units,
                   dropout=dropout)(dgt, is_training=is_training)

    return forward_fn


ffn = build_forward_fn()

ffn_stats = hk.transform(ft.partial(ffn, is_training=False))

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


@ft.partial(jax.jit, static_argnums=(0, 6))
def ce_loss_fn(forward_fn, params, state, rng, a, b, num_classes: int = 10):
    logits, state = forward_fn(params, state, rng, a)

    labels = jnn.one_hot(b, num_classes=num_classes)
    labels = optax.smooth_labels(labels, 2e-2)

    # Weight decay
    l2_loss = 0.5 * jnp.mean(jnp.array([jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)], dtype=jnp.float32))
    l1_loss = jnp.mean(jnp.array([jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params)], dtype=jnp.float32))

    # Normalized CE loss and Focal loss so it is more smooth and gives back better feedback
    # logits = jnp.clip(logits, a_min=jnp.log(1e-12), a_max=jnp.log(1 - 1e-12))
    ce = -labels * logits


    # CE loss
    ce_loss = jnp.sum(ce, axis=1)
    ce_loss = jnp.mean(ce_loss, axis=0)

    # y_pred = jnp.exp(logits)
    # Focal Loss
    #f_loss = focal_loss(labels, y_pred, ce, 2.0, 4.0) # + focal_loss(labels, y_pred, ce, 3.0, 4.0) # + focal_loss(labels, y_pred, ce, 4.0, 4.0)

    # Double Soft F1 Loss
    # tp = jnp.sum(labels * y_pred, axis=0)
    # fp = jnp.sum((1 - labels) * y_pred, axis=0)
    # fn = jnp.sum(labels * (1 - y_pred), axis=0)
    # tn = jnp.sum((1 - labels) * (1 - y_pred), axis=0)
    # soft_f11 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    # soft_f10 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    # cost1 = 1 - soft_f11
    # cost0 = 1 - soft_f10
    # f1_loss = jnp.mean(0.5 * (cost1 + cost0))

    # soft f1 score loss + focal loss and weight decay and l1 loss
    return ce_loss + 1e-12 * (l2_loss + l1_loss), state


loss_fn = ft.partial(ce_loss_fn, l_apply)

learning_rate = 1e-3
grad_clip_value = 1.0
# scheduler = optax.exponential_decay(init_value=learning_rate, transition_steps=6000, decay_rate=0.99)

optimizer = optax.chain(
    #optax.adaptive_grad_clip(grad_clip_value),
    # optax.sgd(learning_rate=learning_rate, momentum=0.99, nesterov=True),
    #optax.scale_by_radam(b1=0.9, eps=1e-4),
    optax.scale_by_adam(),
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
rng1, rng2, rng3, rng = jr.split(rng, 4)

epoch_gen_temp = process_gen(rng1)
bx, _ = next(epoch_gen_temp)
b = jnp.expand_dims(bx[0, 0, :, :], axis=0)


num_steps, _, params, state, opt_state = updater.init(rng2, b)
# Summarize network parameters
print('Network Summary.......................')
print(hk.experimental.tabulate(ffn_stats)(b))

# Training loop
print("Starting training loop..........................")
num_epochs = 4

upd_fn = updater.update

batch_update = jax.pmap(upd_fn, axis_name='devices', in_axes=0, out_axes=0)

params = jax.device_put_replicated(params, devices=jax.devices())
state = jax.device_put_replicated(state, devices=jax.devices())
opt_state = jax.device_put_replicated(opt_state, devices=jax.devices())
num_steps = jax.device_put_replicated(num_steps, devices=jax.devices())


def replicate_tree(t, num_devices):
    return jax.tree_util.tree_map(lambda x: jnp.array([x] * num_devices), t)


n_devices = jax.local_device_count()

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

btchs = 10
count = xt.shape[0] // btchs

for j in tqdm(range(count)):
    rng, = jr.split(rng, 1)
    a, b = j * btchs, (j + 1) * btchs
    pbt = xt[a:b, :, :, :]
    logits, _ = fast_apply(params, state, rng, pbt, is_training=False)
    res[a:b] = np.array(jnp.argmax(jnp.exp(logits), axis=1), dtype=np.int64)

df = pd.DataFrame({'ImageId': np.arange(1, xt.shape[0] + 1, dtype=np.int64), 'Label': res})

df.to_csv('../data/digits/results.csv', index=False)
