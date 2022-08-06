import functools as ft
import pickle

import jax
import jax.nn as jnn
import jax.random as jr
import jax.numpy as jnp

from jax.scipy.special import logsumexp

import numpy as np

import optax

import einops

import haiku as hk
import haiku.initializers as hki
import pandas as pd

from tqdm.auto import tqdm


class SepConv2d(hk.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="SAME"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x, is_training: bool):
        init = hki.VarianceScaling(1.0)
        x = hk.Conv2D(output_channels=self.in_channels, kernel_shape=self.kernel_size, stride=self.stride, padding=self.padding, w_init=init, b_init=hki.Constant(1e-6), name="depthwise")(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9, scale_init=hki.TruncatedNormal(1e-6, 1.0), offset_init=hki.RandomNormal(1e-6))(x, is_training)
        return hk.Conv2D(output_channels=self.in_channels, kernel_shape=self.kernel_size, name="pointwise")(x)


class FeedForward(hk.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def __call__(self, x, is_training: bool):
        dropout = self.dropout if is_training else 0.0
        init = hki.VarianceScaling(1.0)
        x = hk.Linear(output_size=self.hidden_dim, w_init=init, b_init=hki.Constant(1e-6))(x)
        x = jnn.gelu(x)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = hk.Linear(output_size=self.dim, w_init=init, b_init=hki.Constant(1e-6))(x)
        return hk.dropout(hk.next_rng_key(), dropout, x)


class ConvAttention(hk.Module):
    def __init__(self, dim, img_size, heads=16, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout=0.2, last_stage=False):
        super().__init__()
        self.last_stage = last_stage
        self.img_size = img_size
        self.inner_dim = dim_head * heads
        self.project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.kernel_size = kernel_size
        self.q_stride = q_stride
        self.k_stride = k_stride
        self.v_stride = v_stride
        self.dropout = dropout
        self.dim = dim

    def __call__(self, x, is_training: bool):
        dropout = self.dropout if is_training else 0.0

        init = hki.VarianceScaling(1.0)
        b, n, _ = x.shape
        h = self.heads

        if self.last_stage:
            cls_token = x[:, 0, :]
            x = x[:, 1:, :]
            cls_token = einops.rearrange(jnp.expand_dims(cls_token, axis=1), 'b n (h d) -> b h n d', h=h)
        x = einops.rearrange(x, 'b (l w) n -> b l w n', l=self.img_size, w=self.img_size)

        q = SepConv2d(self.dim, self.inner_dim, kernel_size=self.kernel_size, stride=self.q_stride)(x, is_training)
        q = einops.rearrange(q, 'b l w (h d)  -> b h (l w) d', h=h)

        v = SepConv2d(self.dim, self.inner_dim, kernel_size=self.kernel_size, stride=self.v_stride)(x, is_training)
        v = einops.rearrange(v, 'b l w (h d)  -> b h (l w) d', h=h)

        k = SepConv2d(self.dim, self.inner_dim, kernel_size=self.kernel_size, stride=self.k_stride)(x, is_training)
        k = einops.rearrange(k, 'b l w (h d) -> b h (l w) d', h=h)

        if self.last_stage:
            q = jnp.concatenate((cls_token, q), axis=2)
            v = jnp.concatenate((cls_token, v), axis=2)
            k = jnp.concatenate((cls_token, k), axis=2)

        dots = jnp.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = jnn.softmax(dots, axis=-1)

        out = jnp.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        if not self.project_out:
            return out
        out = hk.Linear(self.dim, w_init=init, b_init=hki.Constant(0.0))(out)
        return hk.dropout(hk.next_rng_key(), dropout, out)


class TransformerStage(hk.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0.5, last_stage=False):
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.last_stage = last_stage

    def __call__(self, x, is_training: bool):
        for i in range(self.depth):
            a = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, scale_init=hki.Constant(1.0), offset_init=hki.Constant(0.0))(ConvAttention(self.dim, self.img_size, heads=self.heads, dropout=self.dropout, last_stage=self.last_stage)(x, is_training))
            x = x + a
            b = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, scale_init=hki.Constant(1.0), offset_init=hki.Constant(0.0))(FeedForward(dim=self.dim, hidden_dim=self.mlp_dim, dropout=self.dropout)(x, is_training))
            x = x + b
        return x


class CvTransformer(hk.Module):
    def __init__(self, image_size, dim=32, kernels=(3, 3, 3, 2), strides=(2, 2, 2, 2), heads=(2, 4, 8, 16), depth=(2, 4, 8, 16), pool='cls', dropout=0.3, emb_dropout=0.1, scale_dim=2):
        super().__init__("transformer")
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = dim
        self.image_size = image_size
        self.num_classes = 10
        self.kernels = kernels
        self.strides = strides
        self.heads = heads
        self.depth = depth
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.scale_dim = scale_dim

    def __call__(self, img, is_training):
        init = hki.VarianceScaling(1.0)
        #### Stage 1 ####
        xs = hk.Conv2D(self.dim, self.kernels[0], self.strides[0], w_init=init, b_init=hki.Constant(0))(img)
        xs = einops.rearrange(xs, 'b h w c -> b (h w) c', h=self.image_size//2, w=self.image_size//2)
        xs = hk.LayerNorm(-1, create_scale=True, create_offset=True, scale_init=hki.Constant(1.0), offset_init=hki.Constant(0.0))(xs)
        xs = TransformerStage(dim=self.dim, img_size=self.image_size//2, depth=self.depth[0], heads=self.heads[0], dim_head=self.dim, mlp_dim=self.dim * self.scale_dim, dropout=self.dropout)(xs, is_training)
        xs = einops.rearrange(xs, 'b (h w) c -> b h w c', h=self.image_size//2, w=self.image_size//2)

        ##### Stage 2 ####
        scale = self.heads[1] // self.heads[0]
        dim = self.dim * scale
        xs = hk.Conv2D(dim, self.kernels[1], self.strides[1], w_init=init, b_init=hki.Constant(0))(xs)
        xs = einops.rearrange(xs, 'b h w c -> b (h w) c', h=self.image_size // 4, w=self.image_size // 4)
        xs = hk.LayerNorm(-1, create_scale=True, create_offset=True, scale_init=hki.Constant(1.0), offset_init=hki.Constant(0.0))(xs)
        xs = TransformerStage(dim=dim, img_size=self.image_size // 4, depth=self.depth[1], heads=self.heads[1],
                             dim_head=self.dim, mlp_dim=dim * self.scale_dim, dropout=self.dropout)(xs, is_training)
        xs = einops.rearrange(xs, 'b (h w) c -> b h w c', h=self.image_size // 4, w=self.image_size // 4)

        ##### Stage 3 ####
        scale = self.heads[2] // self.heads[1]
        dim = self.dim * scale
        xs = hk.Conv2D(dim, self.kernels[2], self.strides[2], w_init=init, b_init=hki.Constant(0))(xs)
        xs = einops.rearrange(xs, 'b h w c -> b (h w) c', h=self.image_size // 8, w=self.image_size // 8)
        xs = hk.LayerNorm(-1, create_scale=True, create_offset=True, scale_init=hki.Constant(1.0),
                          offset_init=hki.Constant(0.0))(xs)
        xs = TransformerStage(dim=dim, img_size=self.image_size // 8, depth=self.depth[1], heads=self.heads[1],
                              dim_head=self.dim, mlp_dim=dim * self.scale_dim, dropout=self.dropout)(xs, is_training)
        xs = einops.rearrange(xs, 'b (h w) c -> b h w c', h=self.image_size // 8, w=self.image_size // 8)

        ###### Stage 4 ######
        scale = self.heads[3] // self.heads[2]
        dim = scale * dim
        xs = hk.Conv2D(dim, self.kernels[3], self.strides[3], w_init=init, b_init=hki.Constant(0))(xs)
        xs = einops.rearrange(xs, 'b h w c -> b (h w) c', h=self.image_size // 16, w=self.image_size // 16)
        xs = hk.LayerNorm(-1, create_scale=True, create_offset=True, scale_init=hki.Constant(1.0), offset_init=hki.Constant(0.0))(xs)

        b, n, _ = xs.shape

        cls_t = hk.get_parameter('cls_tokens', (1, 1, dim), init=hki.RandomNormal())
        cls_tokens = einops.repeat(cls_t, '() n d -> b n d', b=b)
        xs = jnp.concatenate((cls_tokens, xs), axis=1)

        xs = TransformerStage(dim=dim, img_size=self.image_size // 16, depth=self.depth[2],
                              heads=self.heads[2],
                              dim_head=self.dim, mlp_dim=dim * self.scale_dim,
                              dropout=self.dropout, last_stage=True)(xs, is_training)

        xs = jnp.mean(xs, axis=1) if self.pool == 'mean' else xs[:, 0, :]

        xs = hk.LayerNorm(-1, create_scale=True, create_offset=True, scale_init=hki.Constant(1.0), offset_init=hki.Constant(0.0))(xs)
        out = hk.Linear(self.num_classes)(xs)
        return out - logsumexp(out, axis=1, keepdims=True)

# Load dataset
with open('../data/digits/data2.dict', 'rb') as f:
    data = pickle.load(f)

x = data['train']
y = data['labels']
xt = data['test']


def process_epoch_gen(a, b, batch_size, num_devices):

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


batch_size = 8

process_gen = process_epoch_gen(x, y, batch_size, jax.local_device_count())


def build_forward_fn(image_size=32):
    def forward_fn(dgt: jnp.ndarray, *, is_training: bool) -> jnp.ndarray:
        return CvTransformer(image_size)(dgt, is_training=is_training)

    return forward_fn


ffn = build_forward_fn()

ffn = hk.transform_with_state(ffn)

apply = ffn.apply

l_apply = ft.partial(apply, is_training=True)
l_apply = jax.jit(l_apply)

t_apply = ft.partial(apply, is_training=False)
t_apply = jax.jit(t_apply)

rng = jr.PRNGKey(111)


def focal_loss(labels, y_pred, ce, gamma, alpha):
    weight = labels * jnp.power(1 - y_pred, gamma)
    f_loss = alpha * (weight * ce)
    f_loss = jnp.sum(f_loss, axis=1)
    f_loss = jnp.mean(f_loss, axis=0)
    return f_loss


@ft.partial(jax.jit, static_argnums=(0, 6))
def ce_loss_fn(forward_fn, params, state, rng, a, b, num_classes: int = 10):

    logits, state = forward_fn(params, state, rng, a)

    labels = jnn.one_hot(b, num_classes=num_classes)
    labels = optax.smooth_labels(labels, 2e-2)

    ce = -labels * logits

    # CE loss
    ce_loss = jnp.sum(ce, axis=1)
    ce_loss = jnp.mean(ce_loss, axis=0)

    # Weight decay
    l2_loss = 0.1 * jnp.mean(jnp.array([jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)], dtype=jnp.float32))
    l1_loss = jnp.mean(jnp.array([jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params)], dtype=jnp.float32))

    return ce_loss + 1e-14 * (l2_loss + l1_loss), state


loss_fn = ft.partial(ce_loss_fn, l_apply)

learning_rate = 3e-4
grad_clip_value = 1.0
# scheduler = optax.exponential_decay(init_value=learning_rate, transition_steps=6000, decay_rate=0.99)

optimizer = optax.chain(
    # optax.adaptive_grad_clip(grad_clip_value),
    # optax.sgd(learning_rate=learning_rate, momentum=0.99, nesterov=True),
    # optax.scale_by_radam(b1=0.9, eps=1e-4),
    optax.scale_by_adam(),
    # optax.scale_by_yogi(),
    # optax.scale_by_schedule(scheduler),
    optax.scale(-learning_rate)
)

class ParamsUpdater:
    def __init__(self, net_init, loss, optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss = loss
        self._opt = optimizer

    def init(self, main_rng, x):
        out_rng, init_rng = jax.random.split(main_rng)
        params, state = self._net_init(init_rng, x, is_training=True)
        opt_state = self._opt.init(params)
        return jnp.array(0), out_rng, params, state, opt_state

    def update(self, num_steps, rng, params, state, opt_state, bx: jnp.ndarray, by: jnp.ndarray):
        rng, new_rng = jax.random.split(rng)

        (loss, state), grads = jax.value_and_grad(self._loss, has_aux=True)(params, state, rng, bx, by)

        grads = jax.lax.psum(grads, axis_name='devices')

        updates, opt_state = self._opt.update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)

        metrics = {
            'step': num_steps,
            'loss': loss,
        }

        return num_steps + 1, new_rng, params, state, opt_state, metrics


updater = ParamsUpdater(ffn.init, loss_fn, optimizer)

print("Initializing parameters..........................")
rng1, rng2, rng3, rng = jr.split(rng, 4)

epoch_gen_temp = process_gen(rng1)
bx, _ = next(epoch_gen_temp)
b = jnp.expand_dims(bx[0, 0, :, :], axis=0)


num_steps, _, params, state, opt_state = updater.init(rng2, b)

# Training loop
print("Starting training loop..........................")
num_epochs = 4

upd_fn = updater.update

batch_update = jax.pmap(upd_fn, axis_name='devices', in_axes=0, out_axes=0)

params = jax.device_put_replicated(params, devices=jax.devices())
state = jax.device_put_replicated(state, devices=jax.devices())
opt_state = jax.device_put_replicated(opt_state, devices=jax.devices())
num_steps = jax.device_put_replicated(num_steps, devices=jax.devices())

n_devices = jax.local_device_count()

for i in range(num_epochs):
    rng1, rng2, rng = jr.split(rng, 3)
    rng2 = jax.device_put_replicated(rng2, devices=jax.local_devices())
    for step, (bx, by) in tqdm(enumerate(process_gen(rng1)), total=42000 // batch_size):

        bbx = []
        bby = []
        for k in range(n_devices):
            bbx.append(bx[k])
            bby.append(by[k])

        dbx = jax.device_put_sharded(bbx, devices=jax.local_devices())
        dby = jax.device_put_sharded(bby, devices=jax.local_devices())

        num_steps, rng2, params, state, opt_state, metrics = batch_update(num_steps, rng2, params, state, opt_state, dbx, dby)
        if (step + 1) % 8 == 0:
            print(f"......Epoch {i} | Step {step} | Metrics\n\n{metrics} .....................................")

print("Starting evaluation loop........................")
params = jax.device_get(jax.tree_util.tree_map(lambda g: g[0], params))
state = jax.device_get(jax.tree_util.tree_map(lambda g: g[0], state))

res = np.zeros(xt.shape[0], dtype=np.int64)

btchs = 10
count = xt.shape[0] // btchs

for j in tqdm(range(count)):
    rng, = jr.split(rng, 1)
    a, b = j * btchs, (j + 1) * btchs
    pbt = xt[a:b, :, :, :]
    logits, _ = t_apply(params, state, rng, pbt)
    res[a:b] = np.array(jnp.argmax(jnp.exp(logits), axis=1), dtype=np.int64)

df = pd.DataFrame({'ImageId': np.arange(1, xt.shape[0] + 1, dtype=np.int64), 'Label': res})

df.to_csv('../data/digits/results.csv', index=False)
