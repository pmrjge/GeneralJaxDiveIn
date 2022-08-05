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

    def __call__(self, x, is_training: bool):
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


class TransformerStage(hk.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0.5, last_stage=False):
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.last_stage = last_stage

    def __call__(self, x, is_training: bool):
        for i in range(self.depth):
            x = x + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(ConvAttention(self.dim, self.img_size, heads=self.heads, dropout=self.dropout, last_stage=self.last_stage)(x, is_training))
            x = x + hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(FeedForward(self.dim, self.mlp_dim, self.dropout)(x, is_training))
        return x


class CvTransformer(hk.Module):
    def __init__(self, image_size, dim=64, kernels=(3, 3, 3), strides=(2, 2, 2), heads=(1, 4, 16), depth=(2, 4, 20), pool='cls', dropout=0.5, emb_dropout=0.1, scale_dim=4):
        super().__init__("transformer")
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = dim
        self.image_size = image_size
        self.num_classes = 10
        self.kernels = kernels
        self.strides = strides
        self.heads = heads
        self.depth = depth
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.scale_dim = scale_dim

    def __call__(self, img, is_training):
        #### Stage 1 ####
        xs = hk.Conv2D(self.dim, self.kernels[0], self.strides[0])(img)
        xs = einops.rearrange(xs, 'b c h w -> b (h w) c', h=self.image_size//2, w=self.image_size//2)
        xs = hk.LayerNorm(-1, create_scale=True, create_offset=True)(xs)
        xs = TransformerStage(dim=self.dim, img_size=self.image_size//2, depth=self.depth[0], heads=self.heads[0], dim_head=self.dim, mlp_dim=self.dim * self.scale_dim, dropout=self.dropout)(xs, is_training)
        xs = einops.rearrange(xs, 'b (h w) c -> b c h w', h=self.image_size//2, w=self.image_size//2)

        ##### Stage 2 ####
        xs = hk.Conv2D(self.dim, self.kernels[1], self.strides[1])(xs)
        xs = einops.rearrange(xs, 'b c h w -> b (h w) c', h=self.image_size // 4, w=self.image_size // 4)
        xs = hk.LayerNorm(-1, create_scale=True, create_offset=True)(xs)
        scale = self.heads[1] // self.heads[0]
        dim = self.dim * scale
        xs = TransformerStage(dim=dim, img_size=self.image_size // 4, depth=self.depth[1], heads=self.heads[1],
                             dim_head=self.dim, mlp_dim=dim * self.scale_dim, dropout=self.dropout)(xs, is_training)
        xs = einops.rearrange(xs, 'b (h w) c -> b c h w', h=self.image_size // 4, w=self.image_size // 4)

        ###### Stage 3 ######
        scale = self.heads[2] // self.heads[1]
        dim = scale * dim
        xs = hk.Conv2D(dim, self.kernels[2], self.strides[2])(xs)
        xs = einops.rearrange(xs, 'b c h w -> b (h w) c', h=self.image_size // 8, w=self.image_size // 8)
        xs = hk.LayerNorm(-1, create_scale=True, create_offset=True)(xs)

        b, n, _ = xs.shape

        cls_t = hk.get_parameter('cls_tokens', (1, 1, dim), init=hki.RandomNormal())
        cls_tokens = einops.repeat(cls_t, '() n d -> b n d', b=b)
        xs = jnp.concatenate((cls_tokens, xs), axis=1)

        xs = TransformerStage(dim=dim, img_size=self.image_size // 8, depth=self.depth[2],
                              heads=self.heads[2],
                              dim_head=self.dim, mlp_dim=dim * self.scale_dim,
                              dropout=self.dropout, last_stage=True)(xs, is_training)

        xs = jnp.mean(xs, axis=1) if self.pool == 'mean' else xs[:, 0]

        xs = hk.LayerNorm(-1, create_scale=True, create_offset=True)(xs)
        return hk.Linear(self.num_classes)(xs)




