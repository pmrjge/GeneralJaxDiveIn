import logging
from mimetypes import init
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
import datetime as dt
from itertools import product

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder


def load():
    df_train = pd.read_csv('./data/regression/train.csv')
    df_test = pd.read_csv('./data/regression/test.csv')

    cat_none_var = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", "GarageType"]
    cat_nb_var = ["BsmtExposure", "BsmtFinType2", "BsmtCond", "BsmtQual", "BsmtFinType1"]
    cat_mode_var = ["Electrical", "Functional", "KitchenQual", "Exterior1st", "Exterior2nd", "MSZoning", "SaleType", "MasVnrType"]
    cat_mode = {var: df_train[var].mode()[0] for var in cat_mode_var}

    for df in [df_train, df_test]:
        # Fill with None (because null means no building/object)
        df[cat_none_var] = df[cat_none_var].fillna("None")
        
        # Fill with NB (because null means no basememt)
        df[cat_nb_var] = df[cat_nb_var].fillna("NB")
        
        # Fill other categorical variables with mode
        for var in cat_mode_var:
            df[var].fillna(cat_mode[var], inplace=True)
            
        # Drop variable because no information
        df.drop("Utilities", axis=1, inplace=True)

    mean_LF = df_train.groupby("Street")["LotFrontage"].mean()

    train_null = df_train.isna().sum()
    test_null = df_test.isna().sum()
    missing = pd.DataFrame(
                data=[train_null, train_null/df_train.shape[0]*100,
                        test_null, test_null/df_test.shape[0]*100],
                columns=df_train.columns,
                index=["Train Null", "Train Null (%)", "Test Null", "Test Null (%)"]
            ).T.sort_values(["Train Null", "Test Null"], ascending=False)

    # Filter only columns with missing values
    missing = missing.loc[(missing["Train Null"] > 0) | (missing["Test Null"] > 0)]

    df_missing = df_train[missing.index]
    missing_cat = df_missing.loc[:, df_missing.dtypes == "object"].columns
    missing_num = df_missing.loc[:, df_missing.dtypes != "object"].columns
    num_zero_var = missing_num.drop("LotFrontage")

    for df in [df_train, df_test]:
        df.loc[(df["LotFrontage"].isna()) & (df["Street"] == "Grvl"), "LotFrontage"] = mean_LF["Grvl"]
        df.loc[(df["LotFrontage"].isna()) & (df["Street"] == "Pave"), "LotFrontage"] = mean_LF["Pave"]
        
        for var in num_zero_var:
            df[var].fillna(0, inplace=True)

    cat_var = df_train.loc[:, df_train.dtypes == "object"].nunique() # Get variable names and number of unique values
    num_var = df_train.loc[:, df_train.dtypes != "object"].columns # Get variable names

    cat_var_unique = {var: sorted(df_train[var].unique()) for var in cat_var.index}

    # Add "-" for each values to replace none in the DataFrame (25 is highest len of unique values)
    for key, val in cat_var_unique.items():
        cat_var_unique[key] += ["-" for x in range(25-len(val))]

    ord_var1 = ["ExterCond", "HeatingQC"]
    ord_var1_cat = ["Po", "Fa", "TA", "Gd", "Ex"]

    ord_var2 = ["ExterQual", "KitchenQual"]
    ord_var2_cat = ["Fa", "TA", "Gd", "Ex"]

    ord_var3 = ["FireplaceQu", "GarageQual", "GarageCond"]
    ord_var3_cat = ["None", "Po", "Fa", "TA", "Gd", "Ex"]

    ord_var4 = ["BsmtQual"]
    ord_var4_cat = ["NB", "Fa", "TA", "Gd", "Ex"]

    ord_var5 = ["BsmtCond"]
    ord_var5_cat = ["NB", "Po", "Fa", "TA", "Gd"]

    ord_var6 = ["BsmtExposure"]
    ord_var6_cat = ["NB", "No", "Mn", "Av", "Gd"]

    ord_var7 = ["BsmtFinType1", "BsmtFinType2"]
    ord_var7_cat = ["NB", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]

    # Put all in one array for easier iteration
    ord_var = [ord_var1, ord_var2, ord_var3, ord_var4, ord_var5, ord_var6, ord_var7]
    ord_var_cat = [ord_var1_cat, ord_var2_cat, ord_var3_cat, ord_var4_cat, ord_var5_cat, ord_var6_cat, ord_var7_cat]
    ord_all = ord_var1 + ord_var2 + ord_var3 + ord_var4 + ord_var5 + ord_var6 + ord_var7 

    for i in range(len(ord_var)):
        enc = OrdinalEncoder(categories=[ord_var_cat[i]])
        for var in ord_var[i]:
            df_train[var] = enc.fit_transform(df_train[[var]])
            df_test[var] = enc.transform(df_test[[var]])

    cat_var = cat_var.drop(ord_all)
    onehot_var = cat_var[cat_var < 6].index

    df_train = pd.get_dummies(df_train, prefix=onehot_var, columns=onehot_var)
    df_test = pd.get_dummies(df_test, prefix=onehot_var, columns=onehot_var)

    add_var = [var for var in df_train.columns if var not in df_test.columns]

    # Add new columns in the test data with value of 0
    for var in add_var:
        if var != "SalePrice":
            df_test[var] = 0

    # Reorder test data column so it is the same order as the train data
    df_test = df_test[df_train.columns.drop("SalePrice")]
    from category_encoders import MEstimateEncoder

    cat_var = cat_var.drop(onehot_var)
    X_train = df_train.drop("SalePrice", axis=1)
    y_train = df_train["SalePrice"]

    te = MEstimateEncoder(cols=df_train[cat_var.index.append(pd.Index(["MoSold"]))]) # Add MoSold variable to the encoder
    X_train = te.fit_transform(X_train, y_train)
    df_test = te.transform(df_test)

    df_train = pd.concat([X_train, y_train], axis=1)

    df_train_corr = df_train[num_var].corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
    df_train_corr.drop(df_train_corr.iloc[1::2].index, inplace=True)
    df_train_corr = df_train_corr.drop(df_train_corr[df_train_corr['Correlation Coefficient'] == 1.0].index)

    high_corr = df_train_corr['Correlation Coefficient'] > 0.5
    df_train_corr[high_corr].reset_index(drop=True)
    df_train_corr[high_corr].loc[(df_train_corr["Feature 1"]=="SalePrice") | (df_train_corr["Feature 2"]=="SalePrice")].reset_index(drop=True)
    for df in [df_train, df_test]:
        df["GarAreaPerCar"] = (df["GarageArea"] / df["GarageCars"]).fillna(0)
        df["GrLivAreaPerRoom"] = df["GrLivArea"] / df["TotRmsAbvGrd"]
        df["TotalHouseSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        df["TotalFullBath"] = df["FullBath"] + df["BsmtFullBath"]
        df["TotalHalfBath"] = df["HalfBath"] + df["BsmtHalfBath"]
        df["InitHouseAge"] = df["YrSold"] - df["YearBuilt"]
        df["RemodHouseAge"] = df["InitHouseAge"] - (df["YrSold"] - df["YearRemodAdd"])
        df["IsRemod"] = (df["YearRemodAdd"] - df["YearBuilt"]).apply(lambda x: 1 if x > 0 else 0)
        df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).apply(lambda x: 0 if x > 2000 else x)
        df["IsGarage"] = df["GarageYrBlt"].apply(lambda x: 1 if x > 0 else 0)
        df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
        df["AvgQualCond"] = (df["OverallQual"] + df["OverallCond"]) / 2

    for df in [df_train, df_test]:
        df.drop([
            "GarageArea", "GarageCars", "GrLivArea", 
            "TotRmsAbvGrd", "TotalBsmtSF", "1stFlrSF", 
            "2ndFlrSF", "FullBath", "BsmtFullBath", "HalfBath", 
            "BsmtHalfBath", "YrSold", "YearBuilt", "YearRemodAdd",
            "GarageYrBlt", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
            "ScreenPorch", "OverallQual", "OverallCond"
        ], axis=1, inplace=True)

    X_train = df_train.drop(["Id", "SalePrice"], axis=1)
    y_train = df_train.SalePrice

    X_test = df_test.drop("Id", axis=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_log = np.log10(y_train)

    return jnp.array(np.expand_dims(X_train_scaled, axis=2)), jnp.array(y_train_log), jnp.array(np.expand_dims(X_test_scaled, axis=2)), df_test

def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=1e-6, name=name)(x)

class Time2Vec(hk.Module):
    def __init__(self, kernel_size=1):
        super().__init__()
        self.k = kernel_size

    def __call__(self, inputs):
        ii1 = inputs.shape[1]
        init = hki.RandomUniform(0, 1.0 / ii1)
        bias = hk.get_parameter('wb', shape=(ii1,), init=init) * inputs + hk.get_parameter('bb', shape=(ii1,), init=init)
        wa = hk.get_parameter('wa', shape=(1, ii1, self.k), init=init)
        ba = hk.get_parameter('ba', shape=(1, ii1, self.k), init=init)
        dp = jnp.dot(inputs, wa) + ba
        weights = jnp.sin(dp)

        ret = jnp.concatenate([jnp.expand_dims(bias, axis=-1), weights], -1)
        ret = einops.rearrange(ret, "t b c -> t (b c)")
        return ret


class AttentionBlock(hk.Module):
    def __init__(self, num_heads, head_size, ff_dim=None, dropout=0.0):
        super().__init__()
        if ff_dim is None:
            self.ff_dim = head_size
        else:
            self.ff_dim = ff_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.head_size = head_size

    def __call__(self, inputs, is_training=True):
        dropout = self.dropout if is_training else 0
        out_features = inputs.shape[-1]

        x = hk.MultiHeadAttention(num_heads=self.num_heads, key_size=self.head_size, w_init_scale=1.0)(inputs, inputs, inputs)
        x = hk.BatchNorm(False, False, decay_rate=0.99, eps=1e-6)(x, is_training)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = layer_norm(x)

        x = hk.Conv1D(output_channels=self.ff_dim, kernel_shape=1, padding="same")(x)
        x = hk.BatchNorm(False, False, decay_rate=0.99, eps=1e-6)(x, is_training)
        x = jnn.gelu(x, approximate=False)
        x = hk.Conv1D(output_channels=out_features, kernel_shape=1, padding="same")(x)
        x = hk.BatchNorm(False, False, decay_rate=0.99, eps=1e-6)(x, is_training)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.gelu(x, approximate=False)

        return layer_norm(x + inputs)


class TimeDistributed(hk.Module):
    def __init__(self, module, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.module = module

    def __call__(self, x):
        module = self.module
        if len(x.shape) <= 2:
            return module(x)

        x_reshape = einops.rearrange(x, "b c h -> (b c) h")

        y = module(x_reshape)

        return jnp.where(self.batch_first, jnp.reshape(y, newshape=(x.shape[0], -1, y.shape[-1])), jnp.reshape(y, newshape=(-1, x.shape[1], y.shape[-1])))


class TransformerThunk(hk.Module):
    def __init__(self, time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, num_layers=1, dropout=0):
        super().__init__()
        self.time2vec_dim = time2vec_dim
        if ff_dim is None:
            self.ff_dim = head_size
        else:
            self.ff_dim = ff_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_layers = num_layers

    def __call__(self, inputs, is_training=True):
        dropout = self.dropout if is_training else 0.0
        time2vec = Time2Vec(kernel_size=self.time2vec_dim)
        time_embedding = TimeDistributed(time2vec)(inputs)

        x = jnp.concatenate([inputs, time_embedding], axis=-1)
        
        w_init = hki.VarianceScaling(2.0, mode='fan_in', distribution='truncated_normal')
        for i in range(self.num_layers):
            x = AttentionBlock(num_heads=self.num_heads, head_size=self.head_size, ff_dim=self.ff_dim, dropout=self.dropout)(x, is_training)
        x = jnp.mean(x, axis=-1)
        #x = einops.rearrange(x, 'b h c -> b (h c)')
        x = hk.Linear(256, w_init=w_init, b_init=hki.Constant(1e-6))(x)
        x = jnn.gelu(x, approximate=False)
        x = hk.Linear(128, w_init=w_init, b_init=hki.Constant(1e-6))(x)
        x = jnn.gelu(x, approximate=False)
        x = hk.Linear(64, w_init=w_init, b_init=hki.Constant(1e-6))(x)
        x = jnn.gelu(x, approximate=False)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        return hk.get_parameter('p_scale', shape=(1,)) * jnn.sigmoid(hk.Linear(1, w_init=w_init, b_init=hki.Constant(1e-6))(x)) + hk.get_parameter('p_bias', shape=(1,))

def build_forward_fn(num_layers, time2vec_dim, num_heads, head_size, ff_dim=None, dropout=0.5):
    def forward_fn(x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        tr = TransformerThunk(time2vec_dim, num_heads, head_size, ff_dim, num_layers, dropout)
        return tr(x, is_training)

    return forward_fn
        
     
@ft.partial(jax.jit, static_argnums=(0, 6))
def lm_loss_fn(forward_fn, params, state, rng, x, y, is_training: bool = True) -> jnp.ndarray:
    y_pred, state = forward_fn(params, state, rng, x, is_training)

    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
    return jnp.sqrt(jnp.mean((jnp.square(y - y_pred)))) + 1e-5 * l2_loss, state


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

    def update(self, num_steps, rng, params, state, opt_state, x:jnp.ndarray, y: jnp.ndarray):
        rng, new_rng = jax.random.split(rng)

        (loss, state), grads = jax.value_and_grad(self._loss_fn, has_aux=True)(params, state, rng, x, y)

        #loss = jax.lax.pmean(loss, axis_name='j')

        grads = jax.lax.pmean(grads, axis_name='j')

        updates, opt_state = self._opt.update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)

        metrics = {
            'step': num_steps,
            'loss': loss,
        }

        return num_steps + 1, new_rng, params, state, opt_state, metrics


def get_generator_parallel(x, y, rng_key, batch_size, num_devices):
    def batch_generator():
        n = x.shape[0]
        key = rng_key
        kk = batch_size // num_devices
        while True:
            key, k1 = jax.random.split(key)
            perm = jax.random.choice(k1, n, shape=(batch_size,))
            
            yield x[perm, :].reshape(num_devices, kk, *x.shape[1:]), y[perm].reshape(num_devices, kk, *y.shape[1:])
    return batch_generator()

def replicate_tree(t, num_devices):
    return jax.tree_map(lambda x: jnp.array([x] * num_devices), t)


def main():
    max_steps = 50
    num_heads = 8
    head_size = 128
    num_layers = 4
    dropout_rate = 0.2
    grad_clip_value = 1.0
    learning_rate = 0.01
    time2vec_dim = 31
    batch_size = 512
    
    num_devices = jax.local_device_count()

    print("Num devices :::: ", num_devices)

    x, y, x_test, test_ds = load()

    print("Examples :::: ", x.shape)
    print("Examples :::: ", y.shape)
    print("Testing Examples :::: ", x_test.shape)

    rng1, rng = jr.split(jax.random.PRNGKey(111))
    train_dataset = get_generator_parallel(x, y, rng1, batch_size, num_devices)

    forward_fn = build_forward_fn(num_layers, time2vec_dim, num_heads, head_size, dropout=dropout_rate)

    forward_fn = hk.transform_with_state(forward_fn)

    forward_apply = forward_fn.apply
    loss_fn = ft.partial(lm_loss_fn, forward_apply)


    scheduler = optax.exponential_decay(init_value=learning_rate, transition_steps=4000, decay_rate=0.99)

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
    num_steps, rng, params, state, opt_state = updater.init(rng1, w[0, :, :, :])

    params_multi_device = params
    opt_state_multi_device = opt_state
    num_steps_replicated = num_steps
    rng_replicated = rng
    state_multi_device = state

    fn_update = jax.pmap(updater.update, axis_name='j', in_axes=(None, None, None, None, None, 0, 0), out_axes=(None, None, None, None, None, 0))

    logging.info('Starting train loop ++++++++...')
    for i, (w, z) in zip(range(max_steps), train_dataset):
        if (i + 1) % 1 == 0:
            logging.info(f'Step {i} computing forward-backward pass')
        num_steps_replicated, rng_replicated, params_multi_device, state_multi_device, opt_state_multi_device, metrics = \
            fn_update(num_steps_replicated, rng_replicated, params_multi_device, state_multi_device, opt_state_multi_device, w, z)

        if (i + 1) % 1 == 0:
            logging.info(f'At step {i} the loss is {metrics}')
    
    # Test part of the model
    forward_apply = jax.jit(forward_apply, static_argnames=['is_training'])
    params_reduced = params_multi_device # Reduce parameters for single device
    state_reduced = state_multi_device
    N = x_test.shape[0]
    result = jnp.zeros((N,))
    rng = rng_replicated

    count = N // 100
    for i in range(count):
        if i % 200 == 0:
            print('Computing ', i * 100)
        (rng,) = jr.split(rng, 1)
        a, b = i * 100, (i + 1) * 100
        eli = x_test[a:b, :, :]
        fa, _ = forward_apply(params_reduced, state_reduced, rng,  eli, is_training=False)
        result = result.at[a:b].set(fa[:, 0])

    result = np.array(result)
    y_pred = 10 ** y_pred

    output = pd.DataFrame({'Id': test_ds.Id, 'SalePrice': y_pred})
    output.to_csv('./data/submission1.csv', index=False)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
