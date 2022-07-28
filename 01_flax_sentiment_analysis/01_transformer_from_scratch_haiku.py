import pickle
import functools as ft

import jax
import jax.random as jr
import jax.numpy as jnp
import jax.nn as jnn

import optax

import einops as nps

import numpy as np

import haiku as hk
import haiku.initializers as hki


class SelfAttention(hk.Module):
    def __init__(self, dim):
        self.dim = dim
        self.factor = dim ** -0.5
        super(SelfAttention, self).__init__()

    @staticmethod
    def masked_fill(mask, a, fill):
        return jax.lax.select(mask, jax.lax.broadcast(fill, a.shape), a)

    def __call__(self, x, mask=None):
        assert len(x.shape) == 3, '3D Tensor must be provided'

        w_init = hki.VarianceScaling()
        qkv = hk.Linear(output_size=self.dim * 3, with_bias=False, w_init=w_init, b_init=hki.Constant(0))(x)

        q, k, v = tuple(nps.rearrange(qkv, 'b t (d k) -> k b t d ', k=3))

        scaled_dot_prod = self.factor * jnp.einsum('b i d,b j d -> b i j', q, k)

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:], "Mask needs to have same shape as scaled dot product"
            # Other version might be:
            # scaled_dot_prod = jnp.where(mask, -jnp.inf, scaled_dot_prod)
            scaled_dot_prod = jax.lax.select(mask, jax.lax.broadcast(-jnp.inf, scaled_dot_prod.shape), scaled_dot_prod)

        attention = jnn.softmax(scaled_dot_prod, axis=-1)

        return jnp.einsum('b i j, b j d -> b i d', attention, v)


class MultiHeadSelfAttention(hk.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        super(MultiHeadSelfAttention, self).__init__()
        self.dim_head = dim // heads if dim_head is None else dim_head
        assert self.dim_head * heads == dim, "MultiHeadSelfAttention head dim and num heads don't agree"

        self.factor = self.dim_head ** -0.5

        self.dim = dim
        self.heads = heads

    def __call__(self, x, mask=None):
        assert len(x.shape) == 3

        w_init = hki.VarianceScaling()
        qkv = hk.Linear(output_size=self.dim * 3, with_bias=False, w_init=w_init, b_init=hki.Constant(0))(x)

        q, k, v = tuple(nps.rearrange(qkv, 'b t (d k h) -> k b h t d ', k=3, h=self.heads))

        scaled_dot_prod = jnp.einsum('b h i d, b h j d -> b h i j', q, k) * self.factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[2:]
            scaled_dot_prod = jax.lax.select(mask, jax.lax.broadcast(-jnp.inf, scaled_dot_prod.shape), scaled_dot_prod)

        attention = jnn.softmax(scaled_dot_prod, axis=-1)

        out = jnp.einsum('b h i j, b h j d -> b h i d', attention, v)

        out = nps.rearrange(out, "b h t d -> b t (h d)")

        return hk.Linear(self.dim, with_bias=False, w_init=w_init, b_init=hki.Constant(0))(out)


class LinearBlock(hk.Module):
    def __init__(self, dim, dim_linear_block, dropout):
        self.dim = dim
        self.dim_linear_block = dim_linear_block
        self.dropout = dropout
        super(LinearBlock, self).__init__()

    def __call__(self, x, *, is_training: bool):
        dropout = self.dropout if is_training else 0.0

        w_init = hki.VarianceScaling()
        x = hk.Linear(self.dim_linear_block, w_init=w_init, b_init=hki.Constant(0))(x)
        x = jnn.gelu(x)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = hk.Linear(self.dim, w_init=w_init, b_init=hki.Constant(0))(x)
        return hk.dropout(hk.next_rng_key(), dropout, x)


class TransformerBlock(hk.Module):
    def __init__(self, dim, heads=8, dim_head=None, dim_linear_block=1024, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.dim_linear_block = dim_linear_block
        self.dropout = dropout

    def __call__(self, x, mask=None, *, is_training: bool):

        scale_init = hki.Constant(1.0)
        offset_init = hki.Constant(0.0)
        norm = lambda: hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, scale_init=scale_init, offset_init=offset_init)
        tmha = MultiHeadSelfAttention(self.dim, self.heads, self.dim_head)(x, mask)
        tpx = norm()(tmha + x)

        lout = LinearBlock(self.dim, self.dim_linear_block, self.dropout)(tpx, is_training=is_training)

        return norm()(lout + tpx)


class TransformerEncoder(hk.Module):
    def __init__(self, dim, blocks=6, heads=8, dim_head=None, dim_linear_block=1024, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.dim = dim
        self.blocks = blocks
        self.heads = heads
        self.dim_head = dim_head
        self.dim_linear_block = dim_linear_block
        self.dropout = dropout

    def __call__(self, x, mask=None, *, is_training: bool):

        for _ in range(self.blocks):
            x = TransformerBlock(self.dim, self.heads, self.dim_head, self.dim_linear_block, self.dropout)(x, mask, is_training=is_training)
        return x


class PositionalEncodingSin(hk.Module):
    def __init__(self, dim, dropout=0.1, seq_len=20000):
        super(PositionalEncodingSin, self).__init__()
        self.dropout = dropout
        pe = jnp.zeros((1, seq_len, dim), dtype=jnp.float32)
        position = jnp.expand_dims(jnp.arange(0, seq_len, dtype=jnp.float32), axis=1)
        div_term = jnp.exp(jnp.arange(0, dim, 2).astype(float) * (-jnp.log(jnp.array([10000.0])) / dim))
        pe = pe.at[..., 0::2] = jnp.sin(position * div_term)
        pe = pe.at[..., 1::2] = jnp.cos(position * div_term)

        self.pe = pe

    @staticmethod
    def expand_to_batch(tensor, desired_size):
        tile = desired_size // tensor.shape[0]
        return nps.repeat(tensor, 'b ... -> (b tile) ...', tile=tile)

    def forward(self, x, *, is_training: bool):
        dropout = self.dropout if is_training else 0.0

        batch, seq_tokens, _ = x.shape
        x = x + PositionalEncodingSin.expand_to_batch(self.pe[:, :seq_tokens, :], desired_size=batch)
        return hk.dropout(hk.next_rng_key(), dropout, x)

