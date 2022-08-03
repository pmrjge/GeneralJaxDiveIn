import functools as ft
import pickle
from typing import Callable

import einops
import jax
import jax.random as jr
import jax.nn as jnn
import jax.numpy as jnp
from flax import struct

from jax.scipy.special import logsumexp

import flax
import flax.linen as fnn
import flax.linen.initializers as fli

import numpy as np

from tqdm.auto import tqdm


@struct.dataclass
class TransformerConfig:
    dropout_rate: float = 0.3
    attention_dropout_rate: float = 0.1
    deterministic: bool = False
    kernel_init: Callable = fli.xavier_uniform()
    bias_init: Callable = fli.normal(stddev=1e-6)
    c_kernel_init: Callable = fli.lecun_normal()
    c_bias_init: Callable = fli.normal(stddev=1e-6)
    num_layers: int = 50
    n_heads: int = 16
    embed_dim: int = 512
    qkv_dim: int = 512
    mlp_dim: int = 2048
    patch_size: int = 8
    num_patches: int = 80 // 8
    batch_size: int = 6


def img_to_patch(x, patch_size):
    b, h, w, c = x.shape

    x = np.reshape(x, (b, h // patch_size, patch_size, w // patch_size, patch_size, c))
    x = np.transpose(x, (0, 1, 3, 2, 4, 5))
    x = np.reshape(x, (b, -1, *x.shape[3:]))
    return x


class AttentionBlock(fnn.Module):
    conf: TransformerConfig

    @fnn.compact
    def __call__(self, x):
        inp_x = fnn.LayerNorm()(x)
        attn_out = fnn.MultiHeadDotProductAttention(num_heads=self.conf.n_heads, qkv_features=self.conf.qkv_dim, kernel_init=self.conf.kernel_init,
                                                    dropout_rate=self.conf.attention_dropout_rate, broadcast_dropout=False, decode=False, use_bias=False, deterministic=self.conf.deterministic)(inputs_q=inp_x, inputs_kv=inp_x)
        x = x + fnn.Dropout(self.conf.dropout_rate)(attn_out, deterministic=self.config.deterministic)

        linear_out = fnn.LayerNorm()(x)
        linear_out = fnn.Dense(self.conf.mlp_dim)(linear_out)
        linear_out = fnn.gelu(linear_out)
        linear_out = fnn.Dropout(self.conf.dropout_rate)(linear_out, deterministic=self.conf.deterministic)
        linear_out = fnn.Dense(self.conf.embed_dim)(linear_out)

        x = x + fnn.Dropout(self.conf.dropout_rate)(linear_out, deterministic=self.conf.deterministic)
        return x


class ConvolutionalBase(fnn.Module):
    config: TransformerConfig

    @fnn.compact
    def __call__(self, x):
        conf = self.config
        x = fnn.Conv(32, (3, 3), padding='SAME', kernel_init=conf.c_kernel_init, bias_init=conf.c_bias_init)(x)
        x = fnn.gelu(x)
        x = fnn.max_pool(x, 2, 2, padding='SAME')
        x = fnn.Conv(64, (3, 3), padding='SAME', kernel_init=conf.c_kernel_init, bias_init=conf.c_bias_init)(x)
        x = fnn.gelu(x)
        x = fnn.max_pool(x, 2, 2, padding='SAME')
        x = fnn.Conv(128, (3, 3), padding='SAME', kernel_init=conf.c_kernel_init, bias_init=conf.c_bias_init)(x)
        x = fnn.gelu(x)
        x = fnn.max_pool(x, 2, 2, padding='SAME')
        x = fnn.Conv(256, (3, 3), padding='SAME', kernel_init=conf.c_kernel_init, bias_init=conf.c_bias_init)(x)
        x = fnn.gelu(x)
        x = fnn.max_pool(x, 2, 2, padding='SAME')
        x = fnn.Dropout(conf.dropout_rate)(x, deterministic=conf.deterministic)
        x = einops.rearrange(x, 'b p h w c -> b p (h w c)')
        x = fnn.Dense(256)(x)
        x = fnn.gelu(x)
        x = fnn.Dense(64)(x)
        x = fnn.Dropout(conf.dropout_rate)(x, deterministic=conf.deterministic)
        return x


class Transformer(fnn.Module):
    config: TransformerConfig

    @fnn.compact
    def __call__(self, x):
        config = self.config
        x = img_to_patch(x, config.patch_size)
        b, t, _, _, _ = x.shape
        print(x.shape)

        x = ConvolutionalBase(config)(x)

        x = fnn.Dense(config.embed_dim)(x)

        cls_token = self.param('cls_token', fli.normal(stddev=1.0), (1, 1, config.embed_dim)).repeat(b, axis=0)
        x = jnp.concatenate([cls_token, x], axis=1)
        x = x + self.param('pos_embedding', fli.normal(stddev=1.0), (1, 1+config.num_patches, config.embed_dim))[:, :(t+1)]

        x = fnn.Dropout(config.dropout_rate)(x, deterministic=config.deterministic)

        for i in range(config.num_layers):
            x = AttentionBlock(config)(x)

        out = einops.rearrange(x, 'b t c -> b (t c)')
        out = fnn.LayerNorm()(out)
        out = fnn.Dense(10)(out)
        return out - logsumexp(out, axis=1, keepdims=True)


# Load dataset
with open('../data/digits/data2.dict', 'rb') as f:
    data = pickle.load(f)

x = data['train']
y = data['labels']
xt = data['test']


def process_epoch_gen(a, b, batch_size, patch_size, num_devices):

    topo = batch_size // num_devices

    def epoch_generator(rng):
        n = a.shape[0]
        num_batches = n // batch_size
        key, rng = jr.split(rng)

        perm = jr.permutation(key, n)
        for i in range(num_batches):
            i0 = i * batch_size
            i1 = (i + 1) * batch_size
            subp = perm[i0: i1]
            outx = jnp.array(a[subp], dtype=jnp.float32)
            outy = jnp.array(b[subp], dtype=jnp.int32)
            yield outx.reshape(num_devices, topo, *outx.shape[1:]), outy.reshape(num_devices, topo, *outy.shape[1:])

    return epoch_generator


main_rng = jr.PRNGKey(111)

main_rng, x_rng = jr.split(main_rng)

sx = jr.normal(x_rng, (1, 80, 80, 1))

config = TransformerConfig(deterministic=False)

transformer = Transformer(config=config)

main_rng, init_rng, dropout_init_rng = jr.split(main_rng, 3)

params = transformer.init({'params': init_rng, 'dropout': dropout_init_rng}, sx)['params']

main_rng, dropout_rng = jr.split(main_rng)

out = transformer.apply({'params': params}, sx, rngs={'dropout': dropout_rng})

print('Out:: ', out.shape)



