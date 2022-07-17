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

import imageio
import tifffile
import cv2
from PIL import Image

#train = pd.read_csv('/kaggle/input/hubmap-organ-segmentation/train.csv')
train = pd.read_csv('./data/segmentation/train.csv')
#string_to_retrieve_data = lambda x: f"../input/hubmap-organ-segmentation/train_images/{x}.tiff"
def string_to_retrieve_data(x):
    return f"./data/segmentation/train_images/{x}.tiff"

def rle2mask(mask_rle, shape=(1536,1536)):
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.int32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T 

class TrainLoader:
    def __init__(self):
        self.paths = train["id"].apply(lambda x: string_to_retrieve_data(x)).values.tolist()
        self.img_size = 1536

    def size(self):
        return train.shape[0]

    def get_sample(self, idx):
        path = self.paths[idx]
        image = Image.open(str(path))
        image = image.resize([self.img_size, self.img_size])
        image = np.array(image).astype(float)
        image = (image - image.min()) / (image.max() - image.min())
        

        label = rle2mask(train.rle[idx])

        return image, label


import multiprocessing

batch_size = 8
num_cpus = min([max([1,int(multiprocessing.cpu_count() * 0.8)]), int(batch_size * 0.8)])

tl = TrainLoader()

def compute_el(idx):
    return tl.get_sample(idx)

def get_data(perm):
        pool = multiprocessing.Pool(processes=num_cpus)

        perm = np.array(perm).tolist()
        
        outputs_async = pool.map_async(compute_el, perm)
        pool.close()
        pool.join()
        outputs = outputs_async.get()
        
        x, y = zip(*outputs)

        x = jnp.stack(x, axis=0)

        y = jnp.stack(y, axis=0)

        return x, y


def bgenerator(rng_key, batch_size, num_devices):

    def batch_generator():
        n = tl.size()
        key = rng_key
        kk = batch_size // num_devices
        while True:
            key, k1 = jax.random.split(key)
            perm = jax.random.choice(k1, n, shape=(batch_size,))

            x, y = get_data(perm)

            yield x.reshape(num_devices, kk, *x.shape[1:]), y.reshape(num_devices, kk, *y.shape[1:])

    return batch_generator()

def dice_loss(inputs, gtr, smooth=1e-6):
    s1 = jnp.sum(gtr)
    s2 = jnp.sum(inputs)
    intersect = jnp.sum(gtr * inputs)
    return jnp.mean(1 - ((2 * intersect + smooth) / (s1 + s2 + smooth)))


class ConvSimplifier(hk.Module):
    def __init__(self):
        super().__init__()
        self.bn = lambda: hk.BatchNorm(True, True, 0.98)

    def __call__(self, x, is_training):
        w_init = hki.VarianceScaling(1.0)
        b_init = hki.Constant(1e-6)
        
        x1 = hk.Conv2D(128, 3, 2, padding="SAME", w_init=w_init, b_init=b_init)(x)
        x1 = self.bn()(x1, is_training)
        x1 = jnn.gelu(x1)

        x2 = hk.Conv2D(128, 3, 2, padding="SAME", w_init=w_init, b_init=b_init)(x1)
        x2 = self.bn()(x2, is_training)
        x2 = jnn.gelu(x2)

        x3 = hk.Conv2D(256, 3, 2, padding="SAME", w_init=w_init, b_init=b_init)(x2)
        x3 = self.bn()(x3, is_training)
        x3 = jnn.gelu(x3)

        x4 = hk.Conv2D(256, 3, 2, padding="SAME", w_init=w_init, b_init=b_init)(x3)
        x4 = self.bn()(x4, is_training)
        x4 = jnn.gelu(x4)

        x5 = hk.Conv2D(512, 3, 2, padding="SAME", w_init=w_init, b_init=b_init)(x4)
        x5 = self.bn()(x5, is_training)
        x5 = jnn.gelu(x5)

        x6 = hk.Conv2D(1024, 3, 2, padding="SAME", w_init=w_init, b_init=b_init)(x5)
        x6 = self.bn()(x6, is_training)
        x6 = jnn.gelu(x6)

        x7 = hk.Conv2D(1024, 3, 2, padding="SAME", w_init=w_init, b_init=b_init)(x6)
        x7 = self.bn()(x7, is_training)
        x7 = jnn.gelu(x7)

        x8 = hk.Conv2D(2048, 3, 2, padding="SAME", w_init=w_init, b_init=b_init)(x7)
        x8 = self.bn()(x8, is_training)
        x8 = jnn.gelu(x8)

        return x1, x2, x3, x4, x5, x6, x7, x8


class ConvInverse(hk.Module):
    def __init__(self):
        super().__init__()
        self.bn = lambda: hk.BatchNorm(True, True, 0.99)

    def __call__(self, x, is_training):
        w_init = hki.VarianceScaling(1.0)
        b_init = hki.Constant(1e-6)

        x1, x2, x3, x4, x5, x6, x7, x8 = x

        val = hk.Conv2DTranspose(2048, 3, stride=2, padding="SAME", w_init=w_init, b_init=b_init)(x8)
        zcat = jnp.concatenate([val, x7], axis=3)
        val = hk.Conv2D(output_channels=2048, kernel_shape=1, stride=1, padding="SAME", w_init=w_init, b_init=b_init)(zcat)
        val = self.bn()(val, is_training)
        val = jnn.gelu(val)

        val = hk.Conv2DTranspose(1024, 3, stride=2, padding="SAME", w_init=w_init, b_init=b_init)(val)
        zcat = jnp.concatenate([val, x6], axis=3)
        val = hk.Conv2D(output_channels=1024, kernel_shape=3, stride=1, padding="SAME", w_init=w_init, b_init=b_init)(zcat)
        val = self.bn()(val, is_training)
        val = jnn.gelu(val)

        val = hk.Conv2DTranspose(1024, 3, stride=2, padding="SAME", w_init=w_init, b_init=b_init)(val)
        zcat = jnp.concatenate([val, x5], axis=3)
        val = hk.Conv2D(output_channels=1024, kernel_shape=3, stride=1, padding="SAME", w_init=w_init, b_init=b_init)(zcat)
        val = self.bn()(val, is_training)
        val = jnn.gelu(val)

        val = hk.Conv2DTranspose(512, 3, stride=2, padding="SAME", w_init=w_init, b_init=b_init)(val)
        zcat = jnp.concatenate([val, x4], axis=3)
        val = hk.Conv2D(output_channels=512, kernel_shape=3, stride=1, padding="SAME", w_init=w_init, b_init=b_init)(zcat)
        val = self.bn()(val, is_training)
        val = jnn.gelu(val)

        val = hk.Conv2DTranspose(256, 3, stride=2, padding="SAME", w_init=w_init, b_init=b_init)(val)
        zcat = jnp.concatenate([val, x3], axis=3)
        val = hk.Conv2D(output_channels=256, kernel_shape=3, stride=1, padding="SAME", w_init=w_init, b_init=b_init)(zcat)
        val = self.bn()(val, is_training)
        val = jnn.gelu(val)

        val = hk.Conv2DTranspose(256, 3, stride=2, padding="SAME", w_init=w_init, b_init=b_init)(val)
        zcat = jnp.concatenate([val, x2], axis=3)
        val = hk.Conv2D(output_channels=256, kernel_shape=3, stride=1, padding="SAME", w_init=w_init, b_init=b_init)(zcat)
        val = self.bn()(val, is_training)
        val = jnn.gelu(val)

        val = hk.Conv2DTranspose(128, 3, stride=2, padding="SAME", w_init=w_init, b_init=b_init)(val)
        zcat = jnp.concatenate([val, x1], axis=3)
        val = hk.Conv2D(output_channels=128, kernel_shape=3, stride=1, padding="SAME", w_init=w_init, b_init=b_init)(zcat)
        val = self.bn()(val, is_training)
        val = jnn.gelu(val)

        val = hk.Conv2DTranspose(128, 3, stride=2, padding="SAME", w_init=w_init, b_init=b_init)(val)
        val = hk.Conv2D(output_channels=1, kernel_shape=1, stride=1, padding="SAME", w_init=w_init, b_init=b_init)(val)
        val = self.bn()(val, is_training)

        return val


class SimpleUNet(hk.Module):
    def __init__(self, dropout=0.4, name: Optional[str] = None):
        super().__init__(name)
        self.dropout = dropout

    def __call__(self, x, is_training):
        dropout = self.dropout if is_training else 0.0

        w_init = hki.VarianceScaling(1.0)
        b_init = hki.Constant(1e-6)

        x = ConvSimplifier()(x, is_training)

        mask = ConvInverse()(x, is_training)

        return mask


def build_forward_fn(dropout=0.5):
    def forward_fn(x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        return SimpleUNet(dropout)(x, is_training=is_training)

    return forward_fn

@ft.partial(jax.jit, static_argnums=(0, 6))
def lm_loss_fn(forward_fn, params, state, rng, x, y, is_training: bool = True):
    y_pred, state = forward_fn(params, state, rng, x, is_training)

    l2_loss = 0.1 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
    #return jnp.mean(optax.sigmoid_binary_cross_entropy(y_pred, y)) + dice_loss(jnn.sigmoid(y_pred), y, smooth=1e-6) + 1e-4 * l2_loss, state
    return dice_loss(jnn.sigmoid(y_pred), y, smooth=1e-6) + 1e-4 * l2_loss, state

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


logging.getLogger().setLevel(logging.INFO)
grad_clip_value = 1.0
learning_rate = 0.001
batch_size = 2
dropout = 0.5
max_steps = 1000
num_devices = jax.local_device_count()
rng = jr.PRNGKey(111)

print("Number of training examples :::::: ", tl.size())

rng, rng_key = jr.split(rng)

train_dataset = bgenerator(rng_key, batch_size=batch_size, num_devices=num_devices)


forward_fn = build_forward_fn(dropout)
forward_fn = hk.transform_with_state(forward_fn)

forward_apply = forward_fn.apply
loss_fn = ft.partial(lm_loss_fn, forward_apply)

scheduler = optax.exponential_decay(init_value=learning_rate, transition_steps=1000, decay_rate=0.99)

optimizer = optax.chain(
    optax.adaptive_grad_clip(grad_clip_value),
    #optax.sgd(learning_rate=learning_rate, momentum=0.95, nesterov=True),
    optax.scale_by_radam(),
    #optax.scale_by_adam(),
    optax.scale_by_schedule(scheduler),
    optax.scale(-1.0)
)

updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)

logging.info('Initializing parameters........................')

rng1, rng = jr.split(rng)
x, y = next(train_dataset)

num_steps, rng2, params, state, opt_state = updater.init(rng1, x[0, :, :, :, :])

rng1, rng = jr.split(rng)
params_multi_device = params
opt_state_multi_device = opt_state
num_steps_replicated = replicate_tree(num_steps, num_devices)
rng_replicated = rng1
state_multi_device = state

batch_update = jax.pmap(updater.update, axis_name='j', in_axes=(0, None, None, None, None, 0, 0),
                        out_axes=(0, None, None, None, None, 0))

logging.info('Starting train loop ++++++++...')

for i, (imgs, masks) in zip(range(max_steps), train_dataset):
    if (i + 1) % 2 == 0:
        logging.info(f'Step {i} computing forward-backward pass')
    num_steps_replicated, rng_replicated, params_multi_device, state_multi_device, opt_state_multi_device, metrics = batch_update(
        num_steps_replicated, rng_replicated, params_multi_device, state_multi_device, opt_state_multi_device, imgs, masks)

    if (i + 1) % 2 == 0:
        logging.info(f'At step {i} the loss is {metrics}')
