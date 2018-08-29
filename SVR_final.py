import pandas as pd
import numpy as np
import datetime
import os, sys
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
import tensorflow as tf
import copy
import matplotlib.patches as mpatches
import time
from math import sqrt

from pandas import concat
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from numpy import concatenate

np.random.seed(7)
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from Column_settings import *
from W_settings import *
from pollution_plots import *

import matplotlib.patches as mpatches

folder = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

batch_sizes = []
times_test = []
times_train = []


def plot_lstm_vs_time(batch_sizes, times_test, times_train):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('LSTM Seq2Seq batch sizes vs. Runtime', fontsize=16)
    ax.set_ylabel('Time in seconds', fontsize=16)
    ax.set_xlabel('Batch size', fontsize=16)
    ax.plot(batch_sizes, times_train, label='Train time', marker='x')
    ax.plot(batch_sizes, times_test, label='Test time', marker='x')
    ax.legend()
    plt.show()
    fig.savefig('./SVR/temp_saluel')


RMSE = []
R2 = []

def plot_lstm_vs_rmse(hidden_dims, rmses):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('LSTM Seq2Seq Hidden Dim vs. RMSE', fontsize=16)
    ax.set_ylabel('RMSE', fontsize=16)
    ax.set_xlabel('# of hidden dimensions', fontsize=16)
    ax.plot(hidden_dims, rmses, marker='x')
    plt.axvline(x=30, color='red', linestyle='dashed', linewidth=1)
    plt.show()
    fig.savefig('./SVR/dimension_saluel')


def plot_test(final_preds_expand, test_y_expand, str_legend, folder):
    """
    fig, ax = plt.subplots(figsize=(17,8))
    ax.set_title("Test Predictions vs. Actual For Last Year")
    ax.plot(final_preds_expand, color = 'red', label = 'predicted')
    ax.plot(test_y_expand, color = 'green', label = 'actual')
    plt.legend(loc="upper left")
    plt.show()
    plt.savefig('plt_test.png')
    """
    text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fig_verify = plt.figure(figsize=(17,8))
    plt.plot(final_preds_expand, color="blue")
    plt.plot(test_y_expand, color="orange")
    plt.title('Seq2seq Train : salouel')
    plt.ylabel('Pollution [PM10]')
    plt.xlabel('Daily timestep for 1 year')
    #str_legend = 'station: '+ str(station) + '\nR2 = ' + str(r2_score(test_y_expand, final_preds_expand)) + '\nMSE = ' + str(mse.item())
    #str_data = 'Train :' + begin_date + ' - ' + test_date  + '\nTest :' + begin_test + ' - ' + final_date
    first_legend = plt.legend(['predicted', 'actual data'] , loc='upper left')
    dates_legend = plt.legend([str_legend], loc='upper right')

    #mesures_legend = plt.legend([str_legend], loc='upper right')

    ax = plt.gca().add_artist(first_legend)

    #ay = plt.gca().add_artist(mesures_legend)


    #plt.legend([AnyObject()], [str_legend],
     #          handler_map={AnyObject: AnyObjectHandler()})

    plt.show()
    path_model_regression_verify = "./Seq2seq_juillet/"+ folder + "/Seq2seq_lilleSeq2seq_lille" + text + ".png"

    fig_verify.savefig(path_model_regression_verify)


class AnyObject(object):
    pass


class AnyObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle([x0, y0], width, height, facecolor='red',
                                   edgecolor='black', hatch='xx', lw=3,
                                   transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch


np.random.seed(7)


# changed all , by . in cvs files

def write_results(text):
    # file = open("results.txt", "w")

    now = datetime.datetime.now()

    with open("svr_grid.txt", "a") as file:
        text = now.strftime("%Y-%m-%d %H:%M:%S") + ": " + "\n" + text + "\n" + "\n"
        file.write(text)


def scale_transformer(val, column_const):
    # print(column_const)
    return (val - column_const['min']) / (column_const['max'] - column_const['min'])


def reverse_transformer(val, column_const):
    return val * (column_const['max'] - column_const['min']) + column_const['min']


def reverse_y(val, col_PM10):
    return (val - col_PM10['min']) / (col_PM10['max'] - col_PM10['min'])


salouel_raw_data = pd.read_csv("./propositio/salou.csv", header=None, sep=';', decimal=',')

#salouel_raw_data = pd.read_csv("df_roth.csv", header=None, sep=';', decimal=',')


creil_raw_data = pd.read_csv("./propositio/creil.csv", header=None, sep=';', decimal=',')

cols = ['Date', 'PM10', 'RR', 'TN', 'TX', 'TM',
                     'PMERM', 'FFM', 'UM', 'GLOT', 'DXY']

roth_raw_data = pd.read_csv("./propositio/roth.csv", header=None, sep=';', decimal=',')

arr_row_data = [salouel_raw_data, creil_raw_data, roth_raw_data]

"""
Deleting dates column and making it all floats
"""

def preprocessing_data(df):
    print(df.iloc[0])
    df = df.iloc[1:]
    df.columns = cols
    df = df.drop(["Date"], axis=1)
    dxy = df['DXY']
    df = df.drop(["DXY"], axis=1)
    df = df.astype(float)
    df['DXY'] = dxy
    return df

salouel_data = preprocessing_data(salouel_raw_data)
creil_data = preprocessing_data(creil_raw_data)
roth_data = preprocessing_data(roth_raw_data)


# for col in salouel_data.columns:
# salouel_data[col] = salouel_data[col].apply(scale_transformer, args=(settings_array[col],))


def get_season(dt):
    # it returnts string
    #print('type of dt:' + str(type(dt)))
    #print(dt)
    dt = dt.date()
    if (dt.month < 3):
        return 'H'
    elif (dt.month < 6):
        return 'P'
    elif (dt.month < 9):
        return "E"
    elif (dt.month < 12):
        return "A"
    else:
        return "H"


#separation_date = '31/12/2010'
begin_test = '01/01/2015'
test_date = '31/12/2013'
begin_date = '01/01/2010'
#final_date = '31/12/2015'
validation_date = '31/12/2014'

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    print('in series to suprsd')
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    #print(df.info())
    #print(df.head(1))
    if dropnan:
        df.dropna(inplace=True)
    #print(df.info())
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    #print(agg.info())
    agg.columns = names
    # drop rows with NaN values

    if dropnan:
        agg.dropna(inplace = True)

    return agg


def prepare_data(df_temp, keep_season=False, keep_wind=False):

    train_data = df_temp[df_temp['Date'] <= test_date].drop(["Date"], axis=1).dropna(axis=0, how="any")
    validation_data = df_temp[df_temp['Date'] > test_date].dropna(axis=0, how="any")
    validation_data = validation_data[validation_data['Date']<= validation_date].drop(["Date"], axis=1)
    test_data = df_temp[df_temp['Date'] > validation_date].drop(["Date"], axis=1).dropna(axis=0, how="any")


    season_train = train_data['Season']
    season_test = test_data['Season']

    train_data = train_data.drop("Season", axis=1)
    test_data = test_data.drop("Season", axis=1)

    vent_train = train_data["DXY"]
    vent_test = test_data["DXY"]

    train_data = train_data.drop("DXY", axis=1)
    test_data = test_data.drop("DXY", axis=1)

    if(keep_season):
        train_data["Season"] = season_train
        test_data["Season"] = season_test

    if(keep_wind):
        train_data["DXY"] = vent_train
        test_data["DXY"] = vent_test

    train_data = pd.get_dummies(train_data)
    test_data = pd.get_dummies(test_data)
    #print(test_data.info())
    train_data.to_csv('./data/train.csv')

    test_data.to_csv('./data/test.csv')

    print("Dataframes saved...")
    return train_data, test_data, validation_data



def clean_data(array_raw_data):
    global begin_date
    # concat is first
    conc = True
    df = array_raw_data.iloc[1:]
    df.columns = cols
    df_temp = df.drop(["Date"], axis=1)
    dxy = df_temp['DXY']
    df_temp = df_temp.drop(["DXY"], axis=1)
    df_temp = df_temp.astype(float)
    df_temp['DXY'] = dxy
    df_temp["Date"] = pd.to_datetime(df["Date"])

    # df_temp["Date"] = pd.to_datetime(df["Date"])
    df_temp['Season'] = df_temp["Date"].apply(get_season)
    #df_temp['vent'] = df_temp['DXY'].apply(get_dir)

    return df_temp



n_past = 90
n_future = 10

def gen_sequence(df, n_future):
    X_seq = []
    for i in range(n_past, len(df) - n_future + 1):
        X_seq.append(df.iloc[i - n_past : i, :].values)
    return X_seq


def test_station(data, station):


    dataset = preprocessing_data(data)
    print(dataset.head())
    dataset = dataset.drop(["TX", 'TN', 'DXY'], axis=1)

    values = dataset.values
    #print(values[:, 9])

    values = DataFrame(values).dropna().values

    encoder = LabelEncoder()
    #values[:, 9] = encoder.fit_transform(values[:, 9])
    #values[:, 7] = encoder.fit_transform(values[:, 7])
    # values[:, 9] = encoder.fit_transform(values[:, 9])

    print(DataFrame(values).head())
    reframed = series_to_supervised(values, n_in=1)
    print(reframed.head())
    #reframed.drop(reframed.columns[[-9, -8, -7, -6, -5, -4, -3, -2, -1]], axis=1, inplace=True)
    #reframed.drop(reframed.columns[[-7, -6, -5, -4, -3, -2, -1]], axis=1, inplace=True)
    reframed.drop(reframed.columns[[-6, -5, -4, -3, -2, -1]], axis=1, inplace=True)

    # print("####")
    # print(reframed.head())
    # print("####")

    values = reframed.values
    #values = scaled
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    print("values shape")
    print(values.shape)
    scaled_features = scaler.fit_transform(values[:,:])

    #scaled_features = scaler.fit_transform(values[:,:-1])
    scaled_label = scaler.fit_transform(values[:,-1].reshape(-1,1))
    values = np.column_stack((scaled_features, scaled_label))

    n_train_days = 359 * 3
    test_days = n_train_days + 365
    train = values[:n_train_days, :]
    test = values[test_days:, :]
    validation = values[n_train_days:test_days, :]
    # print("test:")
    # print(test.shape)
    # split into input and outputs
    # features take all values except the var1
    validation_X, validation_Y = validation[:, :-1], validation[:, -1]
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]


    # print("train_X:")
    # print(test_X)

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]))
    validation_X = validation_X.reshape((validation_X.shape[0], validation_X.shape[1]))

    # print("train_X:")
    # print(test_X.shape)
    # print(test_X.info())

    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # design network


    from sklearn.svm import SVR
    print("train x")
    print(train_X.shape)
    x = train_X
    y = train_y
    val_X = validation_X
    regr = SVR(C = 0.9, epsilon = 0.06, kernel = 'rbf', gamma = 0.7,
               tol = 0.001, verbose=False, shrinking=True, max_iter = 10000)

    print('X;')
    print(x.shape)
    regr.fit(x, y)
    data_pred = regr.predict(test_X)
    #data_pred = regr.predict(validation_X)

    y_pred = scaler.inverse_transform(data_pred.reshape(-1,1))
    y_inv = scaler.inverse_transform(test_y.reshape(-1,1))
    #y_inv = scaler.inverse_transform(validation_Y.reshape(-1,1))


    mse = mean_squared_error(y_inv, y_pred)
    rmse = np.sqrt(mse)
    print('Mean Squared Error: {:.4f}'.format(mse))
    print('Root Mean Squared Error: {:.4f}'.format(rmse))

    print('Variance score: {:2f}'.format(r2_score(y_inv, y_pred)))


    def plot_preds_actual(preds, actual, station_name, str_legend):
        text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig_verify = plt.figure(figsize=(17,8))
        plt.plot(preds, color="blue")
        plt.plot(actual, color="orange")
        plt.title('SVR: ' + station_name + ' SV test')
        plt.ylabel('value')
        plt.xlabel('row')
        # str_legend = 'station: '+ str(station) + '\nR2 = ' + str(r2_score(test_y_expand, final_preds_expand)) + '\nMSE = ' + str(mse.item())
        # str_data = 'Train :' + begin_date + ' - ' + test_date  + '\nTest :' + begin_test + ' - ' + final_date
        first_legend = plt.legend(['predicted', 'actual data'], loc='upper left')
        dates_legend = plt.legend([str_legend], loc='upper right')
        ax = plt.gca().add_artist(first_legend)
        path_model_regression_verify = "./Seq2seq_juillet/" + folder + "/Seq2seq_lilleSeq2seq_lille" + text + ".png"
        plt.show()
        fig_verify.savefig('./Images/svr__' +station_name )

    str_legend = 'R2: ' + str(r2_score(y_inv, y_pred)) + "\nRMSE: " + str(rmse) + "\nMSE: " + str(mse) + "\nGamma:0.7; C:0.9"

    divider = - 329

    y_real = y_inv[divider:,]

    #np.savetxt("./Data/y_svr.csv", y_real, delimiter=",")


    plot_preds_actual(y_pred[divider:,], y_inv[divider:,], station, str_legend)

    RMSE.append(rmse)
    R2.append(r2_score(y_inv, y_pred))
    res = "Gamma: " + str(0.7) + " #  C: " + str(0.9) +  " #  RMSE: " + str(rmse)  + " #  R2: " + str(r2_score(y_inv, y_pred))

    write_results(res)
datasets =  {'salouel': salouel_raw_data, 'roth': roth_raw_data, 'creil': creil_raw_data}


test_station(roth_raw_data, "Roth")

"""
for g in gamma:
    for c in Cs:
        test_station(roth, "roth", g, c)

print(RMSE)
print(R2)


#test_station(creil_raw_data, 'creil')

for key in datasets.keys():
    test_data(datasets[key], key)

LSTM_rmse = [9.763280406492743, 10.708318892382756, 11.153834756838132, 12.306218908923043, 12.103302673593983, 14.671574557373864]

def plot_lstm_vs_svr(LAGS, lstm_rmse, svr_rmse):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_title('Comparison of LSTM Sequence to Scalar VS SVR Sequence to Scalar', fontsize=16)
    ax.set_ylabel('Time in seconds', fontsize=16)
    ax.set_xlabel('RMSE', fontsize=16)
    ax.plot(LAGS, lstm_rmse, label='LSTM', marker='x')
    ax.plot(LAGS, svr_rmse, label='SVR', marker='x')
    ax.legend()
    plt.show()
    fig.savefig('./SVR/lstm_vs_svr_saluel')
#plot_lstm_vs_svr(LAGS, LSTM_rmse, RMSE)
"""