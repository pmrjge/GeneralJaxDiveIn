import logging
from mimetypes import init
import pickle
from typing import Optional, Mapping, Any

import numpy as np
import pandas as pd
import datetime as dt
from itertools import product
import sklearn
from sklearn.model_selection import RepeatedKFold, cross_val_score

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor



def load_time(f1='./data/series/sales_train.csv', f2='./data/series/test.csv'):
    train_ds = pd.read_csv(f1)
    test_ds = pd.read_csv(f2)
    monthly_data = train_ds.pivot_table(index = ['shop_id','item_id'], values = ['item_cnt_day'], columns = ['date_block_num'], fill_value = 0, aggfunc='sum')
    monthly_data.reset_index(inplace = True)
    train_data = monthly_data.drop(columns= ['shop_id','item_id'], level=0)
    train_data.fillna(0,inplace = True)

    y_train = train_data.values[:,-1:].clip(0, 20)

    sc = StandardScaler()
    x_train = sc.fit_transform(train_data.values[:,:-1])

    test_rows = monthly_data.merge(test_ds, on = ['item_id','shop_id'], how = 'right')
    x_test = test_rows.drop(test_rows.columns[:5], axis=1).drop('ID', axis=1)
    x_test.fillna(0,inplace = True)

    x_test = sc.transform(x_test)

    return x_train, y_train, x_test, test_ds

def main():
    x, y, x_test, test_ds = load_time()

    print("Examples :::: ", x.shape)
    print("Examples :::: ", y.shape)
    print("Testing Examples :::: ", x_test.shape)

    logging.info(f"Fitting boosting algorithm.................")

    model = XGBRegressor(n_estimators=4096, max_depth=4, subsample=0.8, colsample_bytree=0.8)
    #cv = RepeatedKFold(n_splits=4, n_repeats=2, random_state=1)
    #print(sklearn.metrics.get_scorer_names())

    #scores = cross_val_score(model, x, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)

    #scores = np.absolute(scores)

    #logging.info('Fit results.........................')
    #print('Mean Sqrt Square Error %.3f (%.3f)' % (scores.mean(), scores.std()))

    model.fit(x, y)

    logging.info(f"Computing predictions...................")
    result = model.predict(x_test)

    output = pd.DataFrame({'ID': test_ds['ID'], 'item_cnt_month': result.clip(0, 20).ravel()})
    output.to_csv('./data/submission_series.csv', index=False)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
