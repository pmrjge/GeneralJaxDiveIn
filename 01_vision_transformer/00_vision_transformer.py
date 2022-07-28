import functools as ft
import pickle
import os

import jax
import jax.nn as jnn
import jax.random as jr
import jax.numpy as jnp

import optax

import einops

import haiku as hk
import haiku.initializers as hki

import tensorflow as tf

tf.config.experimental.set_visible_devices([], "GPU")


class PreProcessPatches:
    def __init__(self, patch_size=8):
        self.patch_size = patch_size

    def __call__(self, images):
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
        for units in range(self.hidden_units):
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
            x3 = MLP([self.transformer_units_1, self.transformer_units_2], self.dropout)(x3, is_training=is_training)

            encoded_patches = x3 + x2

        representation = self.norm()(encoded_patches)
        representation = einops.rearrange(representation, 'b h t -> b (h t)')
        representation = hk.dropout(hk.next_rng_key(), dropout, representation)

        features = MLP(self.mlp_head_units, self.dropout)(representation, is_training=is_training)

        logits = hk.Linear(10)(features)

        return logits


# Load dataset
with open('../data/digits/data.dict', 'rb') as f:
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


