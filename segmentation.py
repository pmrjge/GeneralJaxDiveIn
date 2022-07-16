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

def rle_to_mask(rle, width, target_size=None):
    if target_size == None:
        target_size = width

    rle = np.array(list(map(int, rle.split())))
    label = np.zeros((width*width))
    
    for start, end in zip(rle[::2], rle[1::2]):
        label[start:start+end] = 1
        
    label = Image.fromarray(label.reshape(width, width))

    label = label.resize((target_size, target_size))
    label = np.array(label).astype(float)

    label = np.round((label - label.min())/(label.max() - label.min()))
    
    return label.T

def mask_to_rle(mask, orig_dim=160):

    size = int(len(mask.flatten())**.5)
    n = Image.fromarray(mask.reshape((size, size)))
    n = n.resize((orig_dim, orig_dim))
    n = np.array(n).astype(np.float32)

    pixels = n.T.flatten()

    pixels = (pixels-min(pixels) > ((max(pixels)-min(pixels))/2)).astype(int)
    
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)

class TrainLoader:
    def __init__(self):
        self.paths = train["id"].apply(lambda x: f"../input/hubmap-organ-segmentation/train_images/{x}.tiff").values.tolist()
        self.img_size = 3000

    def size(self):
        return train.shape[0]

    def get_sample(self, idx):
        path = self.paths[idx]
        image = Image.open(str(path))
        image = image.resize((self.img_size, self.img_size))
        image = np.array(image).astype(float)
        image = (image - image.min()) / (image.max() - image.min())
        image = jnp.array(image)

        organ = jnp.array([train.organ.iloc8idx])

        label = rle_to_mask(train.rle[idx], train.img_width[idx], self.img_size)
        label = jnp.array(label)

        return image, organ, label
