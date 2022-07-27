from typing import Callable

import flax
import jax
import jax.random as jr
import flax.linen as fnn
import flax.linen.initializers as fi
from flax.core import freeze, unfreeze
import jax.nn as jnn
import numpy as np


class SelfAttention(fnn.Module):
    embed_size: int
    n_heads: int
    w_init: Callable = fi.lecun_normal()
    b_init: Callable = fi.zeros

    @fnn.compact
    def __call__(self, v, k, q, mask, train=True):
        # Compute head dimension and guarantee embed_size is divisible by n_heads to split the input
        head_dim = self.embed_size // self.n_heads
        assert (head_dim * self.n_heads == self.embed_size)

        # Determine layers
        values = fnn.Dense(head_dim, use_bias=False, kernel_init=self.w_init)
        keys = fnn.Dense(head_dim, use_bias=False, kernel_init=self.w_init)
        query = fnn.Dense(head_dim, use_bias=False, kernel_init=self.w_init)
        fc_out = fnn.Dense(self.embed_size, use_bias=False, kernel_init=self.w_init)

        n = q.shape[0]
        value_len, key_len, query_len = v.shape[1], k.shape[1], q.shape[1]
        # Split embeddings into self.heads pieces

        v = v.reshape(n, value_len, self.n_heads, self.head_dim)
        k = k.reshape(n, key_len, self.n_heads, self.head_dim)
        q = q.reshape(n, query_len, self.n_heads, self.head_dim)




