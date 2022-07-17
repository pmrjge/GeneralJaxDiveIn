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

train = pd.read_csv('/kaggle/input/hubmap-organ-segmentation/train.csv')

def get_mask(image_id, size):
    row = train.loc[train['id'] == image_id].squeeze()
    h, w = row[['img_height', 'img_width']]
    mask = np.zeros(shape=[h * w], dtype=np.int32)
    s = row['rle'].split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        mask[lo : hi] = 1
    mask = mask.reshape([h, w]).T

    mask = cv2.resize(mask, [size, size], interpolation=cv2.INTER_CUBIC).astype(np.int32)

    mask = np.expand_dims(mask, axis=2)
        
    return mask

class TrainLoader:
    def __init__(self):
        self.paths = train["id"].apply(lambda x: f"../input/hubmap-organ-segmentation/train_images/{x}.tiff").values.tolist()
        self.img_size = 3000

    def size(self):
        return train.shape[0]

    def get_sample(self, idx):
        path = self.paths[idx]
        image = Image.open(str(path))
        image = cv2.resize(image, [self.img_size, self.img_size], interpolation=cv2.INTER_CUBIC).astype(np.int32)
        image = np.array(image).astype(float)
        image = (image - image.min()) / (image.max() - image.min())
        image = jnp.array(image)

        label = get_mask(idx, self.img_size)
        label = jnp.array(label)

        return image, label


import multiprocessing

num_cpus = max([1,int(multiprocessing.cpu_count() * 0.7)])


def get_generator_parallel(rng_key, batch_size, num_devices):


    def get_data(tl, perm):
        pool = multiprocessing.Pool(processes=num_cpus)

        perm = np.array(perm).tolist()
        m = lambda p: tl.get_sample(p)

        outputs_async = pool.map_async(m, perm)
        pool.close()
        pool.join()
        outputs = outputs_async.get()
        
        x, o, y = zip(*outputs)

        x = jnp.vstack(x, dtype=jnp.float32)

        o = jnp.vstack(o, dtype=jnp.int32)

        y = jnp.vstack(y)

        return x, o, y

    def batch_generator():
        n = x.shape[0]
        key = rng_key
        kk = batch_size // num_devices
        tl = TrainLoader()
        while True:
            key, k1 = jax.random.split(key)
            perm = jax.random.choice(k1, n, shape=(batch_size,))

            x, o, y = get_data(tl, perm)

            yield x.reshape(num_devices, kk, *x.shape[1:]), y.reshape(num_devices, kk, *y.shape[1:])

    return batch_generator()