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

def rle2mask(mask_rle, shape=(3000,3000)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

class TrainLoader:
    def __init__(self):
        self.paths = train["id"].apply(lambda x: string_to_retrieve_data(x)).values.tolist()
        self.img_size = 3000

    def size(self):
        return train.shape[0]

    def get_sample(self, idx):
        path = self.paths[idx]
        image = Image.open(str(path))
        image = image.resize([self.img_size, self.img_size])
        image = np.array(image).astype(float)
        image = (image - image.min()) / (image.max() - image.min())
        image =  (image - 128.0) / 255.0

        label = rle2mask(train.rle[idx])

        return image, label


import multiprocessing

batch_size = 8
num_cpus = min([max([1,int(multiprocessing.cpu_count() * 0.7)]), int(batch_size * 0.8)])

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


rng = jr.PRNGKey(0)

rng_key, rng = jr.split(rng)
num_devices = jax.local_device_count()

generator = bgenerator(rng_key, batch_size, num_devices)

data = next(generator)

x, y = data

print(x.shape)
print(y.shape)


