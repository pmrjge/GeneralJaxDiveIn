import functools as ft
import pickle
import os

import jax
import jax.nn as jnn
import jax.random as jr
import jax.numpy as jnp
import numpy as np

import optax

import einops

import haiku as hk
import haiku.initializers as hki
import pandas as pd

import tensorflow as tf
from tqdm import tqdm

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
            hk.Embed(vocab_size=self.num_patches, embed_dim=self.projection_dim, w_init=w_init, name="position_embed")(self.positions)


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
    def __init__(self, num_patches=12*12, projection_dim=1024, num_blocks=8, num_heads=8, transformer_units_1=2048, transformer_units_2=1024, mlp_head_units=(2048, 1024), dropout=0.5):
        super(ViT, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.transformer_units_1 = transformer_units_1
        self.transformer_units_2 = transformer_units_2
        self.mlp_head_units = mlp_head_units
        self.dropout = dropout
        self.norm = lambda: hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=1e-6, scale_init=hki.Constant(1.0), offset_init=hki.Constant(0.0))

    def __call__(self, patches, *, is_training: bool):
        dropout = self.dropout if is_training else 0.0

        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)

        w_init = hki.VarianceScaling()
        for _ in range(self.num_blocks):
            x1 = self.norm()(encoded_patches)
            attention = hk.MultiHeadAttention(self.num_heads, self.projection_dim // self.num_heads, w_init=w_init)(x1, x1, x1)
            x2 = attention + encoded_patches
            x3 = self.norm()(x2)
            x3 = MLP((self.transformer_units_1, self.transformer_units_2), self.dropout)(x3, is_training=is_training)

            encoded_patches = x3 + x2

        representation = self.norm()(encoded_patches)
        representation = einops.rearrange(representation, 'b h t -> b (h t)')
        representation = hk.dropout(hk.next_rng_key(), dropout, representation)

        features = MLP(self.mlp_head_units, self.dropout)(representation, is_training=is_training)

        logits = hk.Linear(10)(features)

        return logits


# Load dataset
with open('../data/digits/data1.dict', 'rb') as f:
    data = pickle.load(f)

x = data['train']
y = data['labels']
xt = data['test']


def process_epoch_gen(a, b, batch_size, patch_size):

    proc = PreProcessPatches(patch_size=patch_size)

    def epoch_generator(rng):
        n = a.shape[0]
        num_batches = n // batch_size
        key, rng = jr.split(rng)

        perm = jr.permutation(key, n)
        for i in range(num_batches):
            i0 = i * batch_size
            i1 = (i+1) * batch_size
            subp = perm[i0: i1]
            yield proc(a[subp]), jnp.array(b[subp], dtype=jnp.int32)

    return epoch_generator


batch_size = 14
patch_size = 8

process_gen = process_epoch_gen(x, y, batch_size, patch_size)

patch_dim = 96 // patch_size


def build_forward_fn(num_patches=patch_dim * patch_dim, projection_dim=1024, num_blocks=12, num_heads=8, transformer_units_1=2048, transformer_units_2=1024, mlp_head_units=(2048, 1024), dropout=0.5):
    def forward_fn(dgt: jnp.ndarray, *, is_training: bool) -> jnp.ndarray:
        return ViT(num_patches=num_patches, projection_dim=projection_dim,
                   num_blocks=num_blocks, num_heads=num_heads, transformer_units_1=transformer_units_1,
                   transformer_units_2=transformer_units_2, mlp_head_units=mlp_head_units,
                   dropout=dropout)(dgt, is_training=is_training)

    return forward_fn


ffn = build_forward_fn()

ffn = hk.transform_with_state(ffn)

apply = ffn.apply
fast_apply = jax.jit(apply, static_argnames=('is_training',))

rng = jr.PRNGKey(0)

@ft.partial(jax.jit, static_argnums=(0, 6, 7))
def ce_loss_fn(forward_fn, params, state, rng, a, b, is_training: bool = True, num_classes:int = 10):
    logits, state = forward_fn(params, state, rng, a, is_training=is_training)

    l2_loss = 0.1 * jnp.sum(jnp.array([jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)], dtype=jnp.float32))
    labels = jnn.one_hot(b, num_classes=num_classes)
    return jnp.mean(optax.softmax_cross_entropy(logits, labels)) + 1e-8 * l2_loss, state


loss_fn = ft.partial(ce_loss_fn, fast_apply)

learning_rate = 3e-4
grad_clip_value = 1.0
#scheduler = optax.exponential_decay(init_value=learning_rate, transition_steps=6000, decay_rate=0.99)

optimizer = optax.chain(
    optax.adaptive_grad_clip(grad_clip_value),
    #optax.sgd(learning_rate=learning_rate, momentum=0.99, nesterov=True),
    #optax.scale_by_radam(),
    optax.scale_by_adam(),
    #optax.scale_by_yogi(),
    #optax.scale_by_schedule(scheduler),
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
bx = jnp.expand_dims(bx, axis=0)

num_steps, rng, params, state, opt_state = updater.init(rng2, bx[0, :, :])

# Training loop
print("Starting training loop..........................")
num_epochs = 4

upd_fn = jax.jit(updater.update)

for i in range(num_epochs):
    rng1, rng2, rng = jr.split(rng, 3)
    for step, (bx, by) in tqdm(enumerate(process_gen(rng1)), total=42000 // batch_size):
        num_steps, rng2, params, state, opt_state, metrics = upd_fn(num_steps, rng2, params, state, opt_state, bx, by)
        if (step + 1) % 28 == 0:
            print(f"......Epoch {i} | Step {step} | Metrics\n\n{metrics} .....................................")

print("Starting evaluation loop........................")

res = np.zeros(xt.shape[0], dtype=np.int64)

bts = 10
count = xt.shape[0] // bts

proc = PreProcessPatches(patch_size=patch_size)

for j in tqdm(range(count)):
    rng, = jr.split(rng, 1)
    a, b = j * bts, (j+1) * bts
    pbt = proc(xt[a:b, :, :, :])
    logits, _ = fast_apply(params, state, rng, pbt, is_training=False)
    res[a:b] = np.array(jnp.argmax(jnn.softmax(logits), axis=1), dtype=np.int64)

df = pd.DataFrame({'ImageId': np.arange(1, xt.shape[0]+1, dtype=np.int64), 'Label': res})

df.to_csv('../data/digits/results.csv', index=False)