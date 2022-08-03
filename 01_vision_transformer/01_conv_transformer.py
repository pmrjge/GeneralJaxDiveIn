import functools as ft

import jax
import jax.random as jr
import jax.nn as jnn

from jax.scipy.special import logsumexp

import flax
import flax.linen as fnn
import flax.linen.initializers as fli

import numpy as np

from tqdm import tqdm

