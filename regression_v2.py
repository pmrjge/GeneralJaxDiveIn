import logging
from mimetypes import init
import pickle
from typing import Optional, Mapping, Any

import numpy as np
import optax
import pandas as pd
import datetime as dt
from itertools import product
import sklearn
from sklearn.model_selection import RepeatedKFold, cross_val_score

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

import haiku.initializers as hki

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
import jax
import jax.random as jr

import functools as ft

def load_boost():
    df_train = pd.read_csv('./data/regression/train.csv')
    df_test = pd.read_csv('./data/regression/test.csv')

    cat_none_var = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", "GarageType"]
    cat_nb_var = ["BsmtExposure", "BsmtFinType2", "BsmtCond", "BsmtQual", "BsmtFinType1"]
    cat_mode_var = ["Electrical", "Functional", "KitchenQual", "Exterior1st", "Exterior2nd", "MSZoning", "SaleType", "MasVnrType"]
    cat_mode = {var: df_train[var].mode()[0] for var in cat_mode_var}

    print(df_train.head(10))

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
    x_train = te.fit_transform(X_train, y_train)
    df_test = te.transform(df_test)

    x_train = x_train.drop("Id", axis=1)
    x_test = df_test.drop("Id", axis=1)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    y_train_log = np.log10(y_train)

    return jnp.array(x_train), jnp.array(y_train_log), jnp.array(x_test), df_test


class Regressor0(hk.Module):
    def __init__(self):
        super(Regressor0, self).__init__()

    def __call__(self, x, is_training):
        dropout = 0.5 if is_training else 0.0
        i0 = hki.VarianceScaling(1.0)
        x = hk.Linear(16, w_init=i0)(x)
        x = jnn.relu(x)
        x = hk.Linear(32, w_init=i0)(x)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.relu(x)
        x = hk.Linear(64, w_init=i0)(x)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.relu(x)
        return hk.Linear(1, w_init=i0)(x)


class Regressor1(hk.Module):
    def __init__(self):
        super(Regressor1, self).__init__()

    def __call__(self, x, is_training):
        dropout = 0.4 if is_training else 0.0
        i0 = hki.VarianceScaling(1.0)
        x = hk.Linear(32, w_init=i0)(x)
        x = jnn.relu(x)
        x = hk.Linear(64, w_init=i0)(x)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.relu(x)
        return hk.Linear(1, w_init=i0)(x)


class Regressor2(hk.Module):
    def __init__(self):
        super(Regressor2, self).__init__()

    def __call__(self, x, is_training):
        dropout = 0.3 if is_training else 0.0
        i0 = hki.VarianceScaling(1.0)
        x = hk.Linear(16, w_init=i0)(x)
        x = jnn.relu(x)
        x = hk.Linear(32, w_init=i0)(x)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.relu(x)
        return hk.Linear(1, w_init=i0)(x)


class Regressor3(hk.Module):
    def __init__(self):
        super(Regressor3, self).__init__()

    def __call__(self, x, is_training):
        dropout = 0.5 if is_training else 0.0
        i0 = hki.VarianceScaling(1.0)
        x = hk.Linear(32, w_init=i0)(x)
        x = jnn.relu(x)
        x = hk.Linear(32, w_init=i0)(x)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.relu(x)
        x = hk.Linear(32, w_init=i0)(x)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.relu(x)
        return hk.Linear(1, w_init=i0)(x)


class Regressor4(hk.Module):
    def __init__(self):
        super(Regressor4, self).__init__()

    def __call__(self, x, is_training):
        dropout = 0.7 if is_training else 0.0
        i0 = hki.VarianceScaling(1.0)
        x = hk.Linear(64, w_init=i0)(x)
        x = jnn.relu(x)
        x = hk.Linear(64, w_init=i0)(x)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.relu(x)
        x = hk.Linear(64, w_init=i0)(x)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.relu(x)
        x = hk.Linear(64, w_init=i0)(x)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.relu(x)
        return hk.Linear(1, w_init=i0)(x)


class SuperRegressor(hk.Module):
    def __init__(self):
        super(SuperRegressor, self).__init__()

    def __call__(self, x, is_training):
        y0 = Regressor0()(x, is_training)
        y1 = Regressor1()(x, is_training)
        y2 = Regressor2()(x, is_training)
        y3 = Regressor3()(x, is_training)
        y4 = Regressor4()(x, is_training)

        p0 = jnp.abs(hk.get_parameter('r0w', shape=(1,), init=hki.Constant(0.3)))
        p1 = jnp.abs(hk.get_parameter('r1w', shape=(1,), init=hki.Constant(0.13)))
        p2 = jnp.abs(hk.get_parameter('r2w', shape=(1,), init=hki.Constant(0.16)))
        p3 = jnp.abs(hk.get_parameter('r3w', shape=(1,), init=hki.Constant(0.25)))
        p4 = jnp.abs(hk.get_parameter('r4w', shape=(1,), init=hki.Constant(0.15)))

        w_sum = p0 + p1 + p2 + p3 + p4

        return (p0 * y0 + p1 * y1 + p2 * y2 + p3 * y3 + p4 * y4) / w_sum


def build_forward_fn():
    def forward_fn(x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        return SuperRegressor()(x, is_training)

    return forward_fn


@ft.partial(jax.jit, static_argnames=('forward_fn',))
def lm_loss_fn(forward_fn, params, rng, x, y):
    y_pred = forward_fn(params, rng, x, is_training=True)

    l2_loss = 0.1 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

    return jnp.sqrt(jnp.mean(jnp.square(y_pred - y))) + 1e-6 * l2_loss


class GradientUpdater:
    def __init__(self, net_init, loss_fn, optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    def init(self, master_rng, x):
        out_rng, init_rng = jax.random.split(master_rng)
        params = self._net_init(init_rng, x)
        opt_state = self._opt.init(params)
        return jnp.array(0), out_rng, params, opt_state

    def update(self, num_steps, rng, params, opt_state, x: jnp.ndarray, y: jnp.ndarray):
        rng, new_rng = jax.random.split(rng)

        loss, grads = jax.value_and_grad(self._loss_fn)(params, rng, x, y)

        updates, opt_state = self._opt.update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)

        metrics = {
            'step': num_steps,
            'loss': loss,
        }

        return num_steps + 1, new_rng, params, opt_state, metrics

def main():
    x, y, x_test, test_ds = load_boost()
    print("SHAPE:::::::::::::::::: ", x.shape)
    print("TEST SHAPE :::::::::::::::\n\n", x_test.shape)

    print("Examples :::: ", x.shape)
    print("Examples :::: ", y.shape)
    print("Testing Examples :::: ", x_test.shape)

    logging.info(f"Fitting algorithm.................")
    rng = jr.PRNGKey(111)
    rng, rng_key = jr.split(rng)
    forward_fn = build_forward_fn()
    forward_fn = hk.transform(forward_fn)

    forward_apply = forward_fn.apply
    loss_fn = ft.partial(lm_loss_fn, forward_apply)

    scheduler = optax.exponential_decay(init_value=0.1, transition_steps=100, decay_rate=0.99)

    optimizer = optax.chain(
        optax.adaptive_grad_clip(1.0),
        optax.scale_by_radam(),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0)
    )

    updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)

    num_steps, rng2, params, opt_state = updater.init(rng_key, x[0, :])
    rng1, rng = jr.split(rng)

    for i in range(186):
        num_steps, rng1, params, opt_state, metrics = updater.update(num_steps, rng1, params, opt_state, x, y)
        print(f"Loss metrics at epoch {i} is {metrics}")

    logging.info(f"Computing predictions...................")
    forward_apply = jax.jit(forward_apply, static_argnames=['is_training'])
    result = forward_apply(params, rng, x_test, is_training=False)
    result = (10 ** result).clip(0, 600000).ravel()

    output = pd.DataFrame({'Id': test_ds['Id'], 'SalePrice': result})
    output.to_csv('./data/submission.csv', index=False)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
