import logging
import pickle
from typing import Optional, Mapping, Any

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import haiku.initializers as hki
import einops
import functools as ft

import numpy as np
import optax
import pandas as pd


def load_dataset():
    train_x = np.load('./data/fashion_mnist/train_x.npy')
    train_y = np.load('./data/fashion_mnist/train_y.npy')
    test = np.load('./data/fashion_mnist/test_x.npy')
    test_y = np.load('./data/fashion_mnist/test_y.npy')

    return jnp.array(train_x), jnp.array(train_y, dtype=jnp.int32), jnp.array(test), jnp.array(test_y, dtype=jnp.int32)


def get_generator_parallel(x, y, rng_key, batch_size, num_devices):
    def batch_generator():
        n = x.shape[0]
        key = rng_key
        kk = batch_size // num_devices
        while True:
            key, k1 = jax.random.split(key)
            perm = jax.random.choice(k1, n, shape=(batch_size,))

            yield x[perm, :, :, :].reshape(num_devices, kk, *x.shape[1:]), y[perm].reshape(num_devices, kk, *y.shape[1:])

    return batch_generator()


class ConvNetHybrid(hk.Module):
    def __init__(self, dropout=0.5):
        super(ConvNetHybrid, self).__init__()
        self.dropout = dropout
        scale_init = hki.Constant(1.0)
        offset_init = hki.Constant(1e-8)
        self.bn = lambda: hk.BatchNorm(True, True, 0.98)

    def __call__(self, inputs, is_training=True):
        dropout = self.dropout if is_training else 0.0
        lc_init = hki.VarianceScaling(1.0, 'fan_in', 'truncated_normal')

        # Input regularizer
        x = hk.Conv2D(output_channels=32, kernel_shape=3, stride=1, padding="SAME", w_init=lc_init)(inputs)
        x = self.bn()(x, is_training)
        x = jnn.gelu(x, approximate=False)
        x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)

        x = hk.Conv2D(output_channels=64, kernel_shape=3, stride=1, padding="SAME", w_init=lc_init,
                      b_init=hki.Constant(1e-6))(x)
        x = self.bn()(x, is_training)
        x = jnn.gelu(x, approximate=False)
        x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)

        x = hk.Conv2D(output_channels=128, kernel_shape=3, stride=1, padding="SAME", w_init=lc_init,
                      b_init=hki.Constant(1e-6))(x)
        x = self.bn()(x, is_training)
        x = jnn.gelu(x, approximate=False)
        x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)

        # x = hk.Conv2D(output_channels=128, kernel_shape=3, stride=1, padding="SAME", w_init=lc_init,
        #               b_init=hki.Constant(1e-6))(x)
        # x = self.bn()(x, is_training)
        # x = jnn.gelu(x, approximate=False)
        # x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)

        x = hk.Conv2D(output_channels=192, kernel_shape=3, stride=1, padding="SAME", w_init=lc_init,
                      b_init=hki.Constant(1e-6))(x)
        x = self.bn()(x, is_training)
        x = jnn.gelu(x, approximate=False)
        x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)

        # x = hk.Conv2D(output_channels=192, kernel_shape=3, stride=1, padding="SAME", w_init=lc_init,
        #               b_init=hki.Constant(1e-6))(x)
        # x = self.bn()(x, is_training)
        # x = jnn.gelu(x, approximate=False)
        # x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)

        x = hk.Conv2D(output_channels=256, kernel_shape=3, stride=1, padding="SAME", w_init=lc_init,
                      b_init=hki.Constant(1e-6))(x)
        x = self.bn()(x, is_training)
        x = jnn.gelu(x, approximate=False)
        x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)


        # x = hk.Conv2D(output_channels=384, kernel_shape=3, stride=1, padding="SAME", w_init=lc_init,
        #               b_init=hki.Constant(1e-6))(x)
        # x = self.bn()(x, is_training)
        # x = jnn.gelu(x, approximate=False)
        # x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)

        # x = hk.Conv2D(output_channels=512, kernel_shape=3, stride=1, padding="SAME", w_init=lc_init,
        #               b_init=hki.Constant(1e-6))(x)
        # x = self.bn()(x, is_training)
        # x = jnn.gelu(x, approximate=False)
        # x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)
        #
        # x = hk.Conv2D(output_channels=128, kernel_shape=3, stride=1, padding="SAME", w_init=lc_init,
        #               b_init=hki.Constant(1e-6))(x)
        # x = self.bn()(x, is_training)
        # x = jnn.gelu(x, approximate=False)
        # x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)

        # x = hk.Conv2D(256, 3, 1, w_init=lc_init, b_init=hki.Constant(1e-6))(x)
        # x = self.bn()(x, is_training)
        # x = jnn.gelu(x, approximate=False)
        # x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)

        # x = hk.Conv2D(512, 3, 1, w_init=lc_init, b_init=hki.Constant(1e-6))(x)
        # x = self.bn()(x, is_training)
        # x = jnn.gelu(x, approximate=False)
        # x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)

        y = jnp.mean(x, axis=(1, 2))

        lc_init = hki.VarianceScaling(1.0, 'fan_in', 'truncated_normal')
        y = hk.Linear(128, w_init=lc_init, b_init=hki.Constant(1e-6))(y)
        y = hk.dropout(hk.next_rng_key(), dropout, y)
        y = jnn.gelu(y, approximate=False)

        y = hk.Linear(10, w_init=lc_init, b_init=hki.Constant(1e-6))(y)

        return y - jnn.logsumexp(y)


def build_forward_fn(dropout=0.5):
    def forward_fn(x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        n = ConvNetHybrid(dropout)

        return n(x, is_training=is_training)

    return forward_fn


@ft.partial(jax.jit, static_argnums=(0, 6, 7))
def lm_loss_fn(forward_fn, params, state, rng, x, y, is_training: bool = True, num_classes: int = 10):
    y_pred, state = forward_fn(params, state, rng, x, is_training)

    l2_loss = 0.1 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
    y_hot = jnn.one_hot(y, num_classes=num_classes)
    return jnp.mean(optax.softmax_cross_entropy(y_pred, y_hot)) + 1e-4 * l2_loss, state


class GradientUpdater:
    def __init__(self, net_init, loss_fn, optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    def init(self, master_rng, x):
        out_rng, init_rng = jax.random.split(master_rng)
        params, state = self._net_init(init_rng, x)
        opt_state = self._opt.init(params)
        return jnp.array(0), out_rng, params, state, opt_state

    def update(self, num_steps, rng, params, state, opt_state, x: jnp.ndarray, y: jnp.ndarray):
        rng, new_rng = jax.random.split(rng)

        (loss, state), grads = jax.value_and_grad(self._loss_fn, has_aux=True)(params, state, rng, x, y)

        grads = jax.lax.pmean(grads, axis_name='j')

        updates, opt_state = self._opt.update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)

        metrics = {
            'step': num_steps,
            'loss': loss,
        }

        return num_steps + 1, new_rng, params, state, opt_state, metrics


def replicate_tree(t, num_devices):
    return jax.tree_map(lambda x: jnp.array([x] * num_devices), t)


# training loop
logging.getLogger().setLevel(logging.INFO)
grad_clip_value = 1.0
learning_rate = 0.001
batch_size = 168
dropout = 0.6
max_steps = 500
num_devices = jax.local_device_count()
rng = jr.PRNGKey(111)

x, y, test, test_labels = load_dataset()

print("Number of training examples :::::: ", x.shape[0])
print("Number of testing examples :::::: ", test.shape[0])

rng, rng_key = jr.split(rng)

train_dataset = get_generator_parallel(x, y, rng_key, batch_size, num_devices)

forward_fn = build_forward_fn(dropout)
forward_fn = hk.transform_with_state(forward_fn)

forward_apply = forward_fn.apply
loss_fn = ft.partial(lm_loss_fn, forward_apply)

scheduler = optax.exponential_decay(init_value=learning_rate, transition_steps=400, decay_rate=0.99)

optimizer = optax.chain(
    optax.adaptive_grad_clip(grad_clip_value),
    #optax.sgd(learning_rate=learning_rate, momentum=0.95, nesterov=True),
    #optax.scale_by_radam(),
    optax.scale_by_adam(),
    optax.scale_by_schedule(scheduler),
    optax.scale(-1.0)
)

updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)

logging.info('Initializing parameters...')

rng1, rng = jr.split(rng)
a = next(train_dataset)
w, z = a
num_steps, rng2, params, state, opt_state = updater.init(rng1, w[0, :, :, :])

rng1, rng = jr.split(rng)
params_multi_device = params
opt_state_multi_device = opt_state
num_steps_replicated = replicate_tree(num_steps, num_devices)
rng_replicated = rng1
state_multi_device = state

batch_update = jax.pmap(updater.update, axis_name='j', in_axes=(0, None, None, None, None, 0, 0),
                        out_axes=(0, None, None, None, None, 0))

logging.info('Starting train loop ++++++++...')
for i, (w, z) in zip(range(max_steps), train_dataset):
    if (i + 1) % 100 == 0:
        logging.info(f'Step {i} computing forward-backward pass')
    num_steps_replicated, rng_replicated, params_multi_device, state_multi_device, opt_state_multi_device, metrics = batch_update(
        num_steps_replicated, rng_replicated, params_multi_device, state_multi_device, opt_state_multi_device, w, z)

    if (i + 1) % 100 == 0:
        logging.info(f'At step {i} the loss is {metrics}')

logging.info('Starting evaluation loop +++++++++++++++')
rng1, rng = jr.split(rng)
state = state_multi_device
rng = rng1
params = params_multi_device

fn = jax.jit(forward_apply, static_argnames=['is_training'])

print("Number of testing examples ::::: ", test.shape[0])

res = np.zeros(test.shape[0], dtype=np.int64)
n1 = test.shape[0]

count = n1 // 100
for j in range(count):
    (rng,) = jr.split(rng, 1)
    a, b = j * 100, (j + 1) * 100
    logits, _ = fn(params, state, rng, test[a:b, :, :, :], is_training=False)
    res[a:b] = np.array(jnp.argmax(jnn.softmax(logits), axis=1), dtype=np.int64)

print("Accuracy :::::::::: ", np.mean(res == np.array(test_labels, dtype=np.int32)) * 100)