import functools as ft
import pickle
import os

import jax
import jax.nn as jnn
import jax.random as jr
import jax.numpy as jnp

import optax

import einops

import haiku as hk
import haiku.initializers as hki

import tensorflow as tf

tf.config.experimental.set_visible_devices([], "GPU")


class PreProcessPatches:
    def __init__(self, patch_size=12):
        self.patch_size = patch_size

    def __call__(self, images):
        batch_size = images.shape[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return jnp.array(patches.numpy())




