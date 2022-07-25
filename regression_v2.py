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

    return jnp.array(X_train_scaled), jnp.array(y_train_log), jnp.array(X_test_scaled), df_test


class Regressor(hk.Module):
    def __init__(self):
        super(Regressor, self).__init__()

    def __call__(self, x, is_training):
        dropout = 0.5 if is_training else 0.0
        x = hk.Linear(32)(x)
        x = jnn.relu(x)
        x = hk.Linear(64)(x)
        x = jnn.relu(x)
        x = hk.Linear(128)(x)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.relu(x)
        return hk.Linear(1)(x)


def build_forward_fn():
    def forward_fn(x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        return Regressor()(x, is_training=is_training)

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

    scheduler = optax.exponential_decay(init_value=0.0006, transition_steps=100, decay_rate=0.99)

    optimizer = optax.chain(
        optax.adaptive_grad_clip(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0)
    )

    updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)

    num_steps, rng2, params, opt_state = updater.init(rng_key, x[0, :])
    rng1, rng = jr.split(rng)

    for i in range(200):
        num_steps, rng1, params, opt_state, metrics = updater.update(num_steps, rng1, params, opt_state, x, y)
        print(f"Loss metrics at epoch {i} is {metrics}")

    logging.info(f"Computing predictions...................")
    forward_apply = jax.jit(forward_apply, static_argnames=['is_training'])
    result = forward_apply(params, rng, x_test, is_training=False)
    result = np.power(10, result).clip(0, 500000).ravel()

    output = pd.DataFrame({'Id': test_ds['Id'], 'SalePrice': result})
    output.to_csv('./data/submission.csv', index=False)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
