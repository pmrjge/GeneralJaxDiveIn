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
    def __init__(self, dim, img_size, heads = 16, dim_head = 512):