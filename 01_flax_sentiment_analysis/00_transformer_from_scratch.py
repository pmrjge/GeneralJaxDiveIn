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
    def __call__(self, v, k, q, mask=None):
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

        energy = jnp.einsum('...qhd,...khd->...hqk', q, k)

        if mask is not None:
            energy = jnp.where(mask == 0, -jnp.inf, energy)

        attention = fnn.softmax(energy / (self.embed_size ** 0.5), axis=3)

        out = jnp.einsum('...hql,...lhd->...qhd', attention, v)
        out = eps.rearrange(out, 'n q h d -> n q (h d)')

        return fc_out(out)


class TransformBlock(fnn.Module):
    embed_size: int
    n_heads: int
    forward_expansion: int
    dropout_rate: float
    w_init: Callable = fi.lecun_normal()
    b_init: Callable = fi.zeros

    @fnn.compact
    def __call__(self, value, key, query, mask, train=True):
        dr = self.dropout_rate if train else 0.0
        attention = SelfAttention(embed_size=self.embed_size, n_heads=self.n_heads)
        norm1 = fnn.LayerNorm(bias_init=fi.zeros, scale_init=fi.ones)
        norm2 = fnn.LayerNorm(bias_init=fi.zeros, scale_init=fi.ones)

        ff = fnn.Sequential([
            fnn.Dense(self.forward_expansion * self.embed_size, kernel_init=self.w_init, bias_init=self.b_init),
            fnn.gelu,
            fnn.Dense(self.embed_size, kernel_init=self.w_init, bias_init=self.b_init),
        ])

        a0 = attention(value, key, query, mask)
        sc = norm1(a0 + query)
        sc = fnn.Dropout(dr)(sc, deterministic=not train)
        f = ff(sc)
        sc1 = norm2(f + sc)
        return fnn.Dropout(dr)(sc1)


class PositionalEmbedding(fnn.Module):
    max_seq_len: int
    embed_size: int

    def setup(self):
        pe = jnp.zeros(self.max_seq_len, self.embed_size)
        for pos in range(self.max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe = pe.at[pos, i].set(jnp.sin(pos / (10000 ** ((2 * i) / self.embed_size))))
                pe = pe.at[pos, i+1].set(jnp.cos(pos / (10000 ** ((2 * (i+1)) / self.embed_size))))

        pe = np.expand_dims(pe, axis=0)
        self.pe = pe

    @fnn.compact
    def __call__(self, x):
        n = x.shape[1]
        return self.pe[:, :n]


class Encoder(fnn.Module):
    vocab_size: int
    embed_size: int
    num_layers: int
    n_heads: int
    forward_expansion: int
    dropout_rate: float
    seq_len: int
    w_init: Callable = fi.lecun_normal()
    b_init: Callable = fi.zeros

    @fnn.compact
    def __call__(self, x, mask, train=True):
        dr = self.dropout_rate if train else 0.0

        # Define layers
        word_e = fnn.Embed(self.vocab_size, self.embed_size)
        pos_e = PositionalEmbedding(max_seq_len=self.seq_len, embed_size=self.embed_size)
        blocks = [TransformBlock(embed_size=self.embed_size, n_heads=self.n_heads, forward_expansion=self.forward_expansion, dropout_rate=self.dropout_rate) for _ in range(self.num_layers)]
        fc = fnn.Sequential([fnn.Dense(256, kernel_init=self.w_init, bias_init=self.b_init), fnn.Dense(128, kernel_init=self.w_init, bias_init=self.b_init), fnn.Dense(2, kernel_init=self.w_init, bias_init=self.b_init)])

        # Compute
        embed_out = word_e(x)
        e = embed_out + pos_e(embed_out)
        e = fnn.Dropout(dr)(e, deterministic=not train)

        for block in blocks:
            e = block(e, e, e, mask, train=train)

        return fc(e)










