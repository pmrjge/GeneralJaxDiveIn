import logging
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
        def norm(): return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, scale_init=scale_init, offset_init=offset_init)
        w_init = hki.VarianceScaling()
        tmha = hk.MultiHeadAttention(self.heads, self.dim // self.heads, w_init=w_init)(x, x, x, mask)
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
        pe = pe.at[..., 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[..., 1::2].set(jnp.cos(position * div_term))

        self.pe = pe

    @staticmethod
    def expand(tensor, desired_size):
        tile = desired_size // tensor.shape[0]
        return nps.repeat(tensor, 'b ... -> (b tile) ...', tile=tile)

    def __call__(self, x, *, is_training: bool):
        dropout = self.dropout if is_training else 0.0

        batch, seq_tokens, _ = x.shape
        x = x + PositionalEncodingSin.expand(self.pe[:, :seq_tokens, :], desired_size=batch)
        return hk.dropout(hk.next_rng_key(), dropout, x)


class Transformer(hk.Module):
    def __init__(self, vocab_size, seq_len, embed_dim=512, num_blocks=6, heads=8, dim_linear_block=1024, dropout=0.1, dim_head=None):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.heads = heads
        self.dim_linear_block = dim_linear_block
        self.dropout = dropout
        self.seq_len = seq_len
        self.dim_head = dim_head
        super(Transformer, self).__init__()

    def __call__(self, x, *, is_training: bool):
        dropout = self.dropout if is_training else 0.0

        emb = hk.Embed(self.vocab_size, self.embed_dim)(x)
        pe = PositionalEncodingSin(self.embed_dim, self.dropout, self.seq_len)(emb, is_training=is_training)
        emb = pe + emb
        emb = hk.dropout(hk.next_rng_key(), dropout, emb)

        t = TransformerEncoder(self.embed_dim, self.num_blocks, self.heads, self.dim_head, self.dim_linear_block, self.dropout)(emb, is_training=is_training)

        w_init = hki.VarianceScaling()
        b_init = hki.Constant(0)
        fc = hk.Sequential([hk.Linear(512, w_init=w_init, b_init=b_init), jnn.gelu,
                            hk.Linear(256, w_init=w_init, b_init=b_init), jnn.gelu,
                            hk.Linear(2, w_init=w_init, b_init=b_init)])

        return fc(t)


def make_generator(x_train, y_train, batch_size):
    n = x_train.shape[0]
    num_batches = n // batch_size

    def generate_epoch(rng_key):
        rng, _ = jr.split(rng_key)

        perm = jr.permutation(rng, n, axis=0)
        for i in range(num_batches):
            i0 = i * batch_size
            i1 = (i+1) * batch_size
            vi = perm[i0:i1]
            yield jnp.array(x_train[vi], dtype=jnp.int32), jnp.array(y_train[vi], dtype=jnp.int32)
    return generate_epoch


# Load data
with open('../data/sentiment_analysis/train_data.dict', 'rb') as f:
    data = pickle.load(f)

xTrain = jnp.expand_dims(jnp.array(data['x_train'], dtype=jnp.int32), axis=2)
yTrain = jnp.array(data['y_train'], dtype=jnp.int32)
xTest = jnp.expand_dims(jnp.array(data['x_test'], dtype=jnp.int32), axis=2)
vocab_count = data['vc']
vocab = data['vocab']

print(xTrain[0, :])


def build_forward_fn(vocab_size, seq_len):
    def forward_fn(x: jnp.ndarray, *, is_training: bool) -> jnp.ndarray:
        return Transformer(vocab_size, seq_len)(x, is_training=is_training)

    return forward_fn


@ft.partial(jax.jit, static_argnums=(0, 6))
def loss_fn(forward_fn, params, state, rng, x, y, *, is_training: bool = True):
    logits, state = forward_fn(params, state, rng, x, is_training=is_training)

    l2_loss = 0.1 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

    labels = hk.one_hot(y, num_classes=2)

    return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels)) + 1e-7 * l2_loss, state


class ParamsUpdater:
    def __init__(self, net_init, loss_fn, optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    def init(self, main_rng, x):
        out_rng, init_rng = jax.random.split(main_rng)
        params, state = self._net_init(init_rng, x, is_training=False)
        opt_state = self._opt.init(params)
        return jnp.array(0), out_rng, params, state, opt_state

    def update(self, num_steps, rng, params, state, opt_state, x: jnp.ndarray, y: jnp.ndarray):
        rng, new_rng = jax.random.split(rng)

        (loss, state), grads = jax.value_and_grad(self._loss_fn, has_aux=True)(params, state, rng, x, y)

        updates, opt_state = self._opt.update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)

        metrics = {
            'step': num_steps,
            'loss': loss,
        }

        return num_steps + 1, new_rng, params, state, opt_state, metrics


grad_clip_value = 1.0
learning_rate = 0.01
batch_size = 100
seq_len = 60

num_epochs = 10

rng = jr.PRNGKey(111)

print('Number of training examples :::::: ', xTrain.shape[0])
print('Number of testing examples :::::: ', xTest.shape[0])

rng, rng_key = jr.split(rng)

train_epoch = make_generator(xTrain, yTrain, batch_size)

# Init network parameters
forward_fn = build_forward_fn(vocab_size=vocab_count, seq_len=seq_len)
forward_fn = hk.transform_with_state(forward_fn)

apply = jax.jit(forward_fn.apply, static_argnames=('is_training',))
loss_f = ft.partial(loss_fn, apply)

scheduler = optax.exponential_decay(init_value=learning_rate, transition_steps=100, decay_rate=0.99)

optimizer = optax.chain(
    optax.adaptive_grad_clip(grad_clip_value),
    #optax.sgd(learning_rate=learning_rate, momentum=0.95, nesterov=True),
    optax.scale_by_radam(),
    #optax.scale_by_adam(),
    optax.scale_by_schedule(scheduler),
    optax.scale(-1.0)
)

updater = ParamsUpdater(forward_fn.init, loss_fn, optimizer)

print("Initializing parameters.........................")

rng, rng1 = jr.split(rng)

i = xTrain[0, :]
num_steps, _, params, state, opt_state = updater.init(rng1, i)

print('Starting train loop >>>>>>>>>>>>>>>>..............>>>>>>>>>>>>>>')

for i in range(num_epochs):
    rng, rngi, rngj = jr.split(rng, 3)
    for x, y in train_epoch(rngi):
        num_steps, rngj, params, state, opt_state, metrics = updater.update(num_steps, rngj, params, state, opt_state, x, y)
        if (i + 1) % 2 == 0:
            print(f'Step {i} computed fb pass.\n\n Loss is {metrics}')

