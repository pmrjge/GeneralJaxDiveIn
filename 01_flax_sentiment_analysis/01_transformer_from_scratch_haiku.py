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