from typing import Callable

import flax
import jax
import jax.random as jr
import jax.numpy as jnp
import flax.linen as fnn
import flax.linen.initializers as fi
from flax.core import freeze, unfreeze
import jax.nn as jnn
import numpy as np
import einops as eps


class SelfAttention(fnn.Module):
    embed_size: int
    n_heads: int
    w_init: Callable = fi.lecun_normal()
    b_init: Callable = fi.zeros

    @fnn.compact
    def __call__(self, v, k, q, mask=None, train=True):
        # Compute head dimension and guarantee embed_size is divisible by n_heads to split the input
        head_dim = self.embed_size // self.n_heads
        assert (head_dim * self.n_heads == self.embed_size)

        # Determine layers
        values = fnn.Dense(head_dim, use_bias=False, kernel_init=self.w_init)
        keys = fnn.Dense(head_dim, use_bias=False, kernel_init=self.w_init)
        query = fnn.Dense(head_dim, use_bias=False, kernel_init=self.w_init)
        fc_out = fnn.Dense(self.embed_size, kernel_init=self.w_init, bias_init=self.b_init)

        n = q.shape[0]
        value_len, key_len, query_len = v.shape[1], k.shape[1], q.shape[1]
        # Split embeddings into self.heads pieces

        v = v.reshape(n, value_len, self.n_heads, self.head_dim)
        k = k.reshape(n, key_len, self.n_heads, self.head_dim)
        q = q.reshape(n, query_len, self.n_heads, self.head_dim)

        v = values(v)
        k = keys(k)
        q = query(q)

        energy = jnp.einsum_path('nqhd,nkhd -> nhqk', [q, k])

        if mask is not None:
            energy = jnp.where(mask == 0, -1e20, energy)

        attention = fnn.softmax(energy / (self.embed_size ** 0.5), axis=3)

        out = jnp.einsum_path('nhql,nlhd -> nqhd', [attention, v])
        out = eps.rearrange(out, 'n q h d -> n q (h d)')

        return fc_out(out)
