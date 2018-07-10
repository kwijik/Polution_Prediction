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

#print(data)
def test_data(dataset, station_name):

    dataset = preprocessing_data(dataset)
    values = dataset.values

    encoder = LabelEncoder()
    values[:,9] = encoder.fit_transform(values[:,9])

    reframed = series_to_supervised(values)



    reframed.drop(reframed.columns[[-9, -8, -7,-6,-5,-4,-3,-2,-1]], axis=1, inplace=True)

    print("####")
    print(reframed.head())
    print("####")

    values = reframed.values
    #values = scaled
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    scaled_features = scaler.fit_transform(values[:,:-1])
    scaled_label = scaler.fit_transform(values[:,-1].reshape(-1,1))
    values = np.column_stack((scaled_features, scaled_label))

    n_train_hours = 365 * 24 + (365 * 48)
    train = values[:n_train_hours, :]
    test = values[n_train_hours:-365*24, :]
    # split into input and outputs
    # features take all values except the var1
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]



    from sklearn.svm import SVR
    print("train x")
    print(train_X.shape)
    x = train_X
    y = train_y

    regr = SVR(C = 2.0, epsilon = 0.1, kernel = 'rbf', gamma = 0.5,
               tol = 0.001, verbose=False, shrinking=True, max_iter = 10000)

    regr.fit(x, y)
    data_pred = regr.predict(x)
    y_pred = scaler.inverse_transform(data_pred.reshape(-1,1))
    y_inv = scaler.inverse_transform(y.reshape(-1,1))

    mse = mean_squared_error(y_inv, y_pred)
    rmse = np.sqrt(mse)
    print('Mean Squared Error: {:.4f}'.format(mse))
    print('Root Mean Squared Error: {:.4f}'.format(rmse))

    print('Variance score: {:2f}'.format(r2_score(y_inv, y_pred)))


    def plot_preds_actual(preds, actual, station_name, str_legend):
        text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig_verify = plt.figure()
        plt.plot(preds, color="blue")
        plt.plot(actual, color="orange")
        plt.title('SVR Train :' + station_name)
        plt.ylabel('value')
        plt.xlabel('row')
        # str_legend = 'station: '+ str(station) + '\nR2 = ' + str(r2_score(test_y_expand, final_preds_expand)) + '\nMSE = ' + str(mse.item())
        # str_data = 'Train :' + begin_date + ' - ' + test_date  + '\nTest :' + begin_test + ' - ' + final_date
        first_legend = plt.legend(['predicted', 'actual data'], loc='upper left')
        dates_legend = plt.legend([str_legend], loc='upper right')
        ax = plt.gca().add_artist(first_legend)
        path_model_regression_verify = "./Seq2seq_juillet/" + folder + "/Seq2seq_lilleSeq2seq_lille" + text + ".png"
        fig_verify.savefig('./SVR/' +station_name + "_" + text)

    str_legend = 'R2: ' + str(r2_score(y_inv, y_pred)) + "\nRMSE: " + str(rmse) + "\nMSE: " + str(mse)
    plot_preds_actual(y_pred[:24*31*1,], y_inv[:24*31*1,], station_name, str_legend)

datasets =  {'salouel': salouel_raw_data, 'roth': roth_raw_data, 'creil': creil_raw_data}

for key in datasets.keys():
    test_data(datasets[key], key)
