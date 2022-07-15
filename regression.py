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
    missing.style.background_gradient('summer_r')

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

    
