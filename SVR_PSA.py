from math import sqrt
from numpy import concatenate
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import pandas as pd
import time
from W_settings import *
import datetime
from sklearn.svm import SVR

import os



cols = ['Date', 'NO2', 'O3', 'PM10', 'RR', 'TN', 'TX', 'TM',
                     'PMERM', 'FFM', 'DXY', 'UM', 'GLOT']

folder = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def preprocessing_data(df):
    df = df.iloc[1:]
    df.columns = cols
    df = df.drop(["Date", 'NO2', 'O3'], axis=1)
    df = df.astype(float)
    df['vent'] = df['DXY'].apply(get_dir)
    df = df.drop(['DXY'], axis=1)
    #df = df.dropna(axis=0, how="any")
    return df


from pandas import read_csv
creil_raw_data = pd.read_csv("Moyennes_J_creil_2005_2015.csv", header=None, sep=';', decimal=',')
roth_raw_data = pd.read_csv("Moyennes_J_roth_2005_2015.csv", header=None, sep=';', decimal=',')
salouel_raw_data = pd.read_csv("Moyennes_J_salouel_2005_2015.csv", header=None, sep=';', decimal=',')


def test_data(dataset, station_name):
    dataset = preprocessing_data(dataset)
    dataset = dataset.drop("vent", 1)
    values = dataset.values

    print(dataset.head())
    print(dataset.info())

    reframed = series_to_supervised(values)

    reframed.drop(reframed.columns[[-9, -8, -7,-6,-5,-4,-3,-2,-1]], axis=1, inplace=True)

    print("####")
    print(reframed.head())
    print(reframed.info())

    print("####")

    values = reframed.values

    print("values:")
    print(values.shape)

    n_hours = 4
    n_features = 9
    # Splitting dataset into dev, train and test

    n_train_hours = 365 * 9  # 2010 - 2012
    n_dev_hours = 365 * 5  # 2013
    n_test_hours = 365 * 12  # 2014 - 2017
    train = values[:n_train_hours, :]
    dev = values[n_train_hours:(n_train_hours + n_dev_hours), :]
    test = values[(n_train_hours + n_dev_hours):, :]

    # split into input and outputs
    n_obs = n_hours * n_features

    print("dev :")
    print(dev.shape)
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    dev_X, dev_y = dev[:, :n_obs], dev[:, -n_features]
    print("dev x:")
    print(dev_X.shape)
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    print(train_X.shape, len(train_X), train_y.shape)
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # Running SVR on dev
    clf = SVR()
    clf.fit(train_X, train_y)

    y_pred = clf.predict(dev_X)

    dev_y_unscaled = (dev_y * (dataset['PM10'].max(axis=0) - dataset['PM10'].min(axis=0))) + dataset[
        'PM10'].min(axis=0)
    y_pred_unscaled = (y_pred * (dataset['PM10'].max(axis=0) - dataset['PM10'].min(axis=0))) + dataset[
        'PM10'].min(axis=0)

    rmse = sqrt(mean_squared_error(dev_y_unscaled, y_pred_unscaled))
    print('Dev RMSE: %.3f' % rmse)
