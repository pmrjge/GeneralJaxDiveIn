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


class SepConv2d(hk.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="SAME"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x, is_training: bool):
        x = hk.Conv2D(output_channels=self.in_channels, kernel_shape=self.kernel_size, stride=self.stride, padding=self.padding, name="depthwise")(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(x, is_training)
        return hk.Conv2D(output_channels=self.in_channels, kernel_shape=self.kernel_size, name="pointwise")(x)


class FeedForward(hk.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def __call__(self, x, is_training: bool):
        dropout = self.dropout if is_training else 0.0
        x = hk.Linear(output_size=self.hidden_dim)(x)
        x = jnn.gelu(x)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = hk.Linear(output_size=self.dim)(x)
        return hk.dropout(hk.next_rng_key(), dropout, x)


class ConvAttention(hk.Module):
    def __init__(self, dim, img_size, heads=16, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout=0.2, last_stage=False):
        super().__init__()
        self.last_stage = last_stage
        self.img_size = img_size
        self.inner_dim = dim_head * heads
        self.project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.kernel_size = kernel_size
        self.q_stride = q_stride
        self.k_stride = k_stride
        self.v_stride = v_stride
        self.dropout = dropout
        self.dim = dim

    def __call__(self, x, is_training):
        dropout = self.dropout if is_training else 0.0

        b, n, _ = x.shape
        h = self.heads

        if self.last_stage:
            cls_token = x[:, 0, :]
            x = x[:, 1:, :]
            cls_token = einops.rearrange(jnp.expand_dims(cls_token, axis=1), 'b n (h d) -> b h n d', h=h)
        x = einops.rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)

        q = SepConv2d(self.dim, self.inner_dim, kernel_size=self.kernel_size, stride=self.q_stride)(x, is_training)
        q = einops.rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        v = SepConv2d(self.dim, self.inner_dim, kernel_size=self.kernel_size, stride=self.v_stride)(x, is_training)
        v = einops.rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        k = SepConv2d(self.dim, self.inner_dim, kernel_size=self.kernel_size, stride=self.k_stride)(x, is_training)
        k = einops.rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)

        if self.last_stage:
            q = jnp.concatenate((cls_token, q), axis=2)
            k = jnp.concatenate((cls_token, k), axis=2)
            k = jnp.concatenate((cls_token, k), axis=2)

        dots = jnp.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = jnn.softmax(dots, axis=-1)

        out = jnp.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        if not self.project_out:
            return out
        out = hk.Linear(self.dim)(out)
        return hk.dropout(hk.next_rng_key(), dropout, out)


