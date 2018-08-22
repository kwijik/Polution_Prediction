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
    #plt.show()
    fig.savefig('./SVR/temp_saluel')


RMSE = []
R2 = []
Batches = []

def plot_lstm_vs_rmse(hidden_dims, rmses):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('LSTM Seq2Seq Hidden Dim vs. RMSE', fontsize=16)
    ax.set_ylabel('RMSE', fontsize=16)
    ax.set_xlabel('# of hidden dimensions', fontsize=16)
    ax.plot(hidden_dims, rmses, marker='x')
    plt.axvline(x=30, color='red', linestyle='dashed', linewidth=1)
    #plt.show()
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

    #plt.show()
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

    with open("grid_creil_lstm3.txt", "a") as file:
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
    #print(df.iloc[0])
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
    #print('in series to suprsd')
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

    #print("Dataframes saved...")
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


def test_station(data, station, tl):

    df_temp = clean_data(data)

    train_data, test_data, validation_data = prepare_data(df_temp, True, True)

    X_train = train_data.loc[:, ['PM10', 'RR', 'TN', 'TX', 'TM', 'PMERM', 'FFM', 'UM', 'GLOT']].values.copy()
    X_test = test_data.loc[:, ['PM10', 'RR', 'TN', 'TX', 'TM', 'PMERM', 'FFM', 'UM', 'GLOT']].values.copy()




    dataset = preprocessing_data(data)
    #dataset = dataset.drop(["TX", 'TN'], axis=1)
    #dataset = dataset.drop(["TM", 'TN', 'GLOT'], axis=1)

    values = dataset.values
    #print(values[:, 7])

    values = DataFrame(values).dropna().values

    encoder = LabelEncoder()
    #values[:, 7] = encoder.fit_transform(values[:, 7])
    values[:, 9] = encoder.fit_transform(values[:, 9])
    #values[:, 6] = encoder.fit_transform(values[:, 6])


    reframed = series_to_supervised(values, n_in=tl)

    reframed.drop(reframed.columns[[-9, -8, -7, -6, -5, -4, -3, -2, -1]], axis=1, inplace=True)
    #reframed.drop(reframed.columns[[-7, -6, -5, -4, -3, -2, -1]], axis=1, inplace=True)
    #reframed.drop(reframed.columns[[-6, -5, -4, -3, -2, -1]], axis=1, inplace=True)

    # print("####")
    # print(reframed.head())
    # print("####")

    values = reframed.values
    # print("values:")
    # print(values.shape)

    ########
    # LSTM #
    ########

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(values[:, :-1])
    scaled_label = scaler.fit_transform(values[:, -1].reshape(-1, 1))
    values = np.column_stack((scaled_features, scaled_label))

    # print("values:")
    # print(values.shape)

    n_train_days = 359 * 3
    test_days = n_train_days+365
    train = values[:n_train_days, :]
    test = values[test_days:, :]
    validation = values[n_train_days:test_days, :]

    # print("test:")
    # print(test.shape)
    # split into input and outputs
    # features take all values except the var1
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    validation_X, validation_Y = validation[:, :-1], validation[:, -1]


    # print("train_X:")
    # print(test_X)

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    validation_X = validation_X.reshape((validation_X.shape[0], 1, validation_X.shape[1]))

    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # design network
    model = Sequential()
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(50, activation="softmax"))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # print(reframed.head())

    start = time.time()

    # fit network
    ###################### Can change Epochs, Batch size here #######################
    history = model.fit(train_X, train_y, epochs=99, batch_size=20, validation_data=(validation_X, validation_Y),
                        verbose=1, shuffle=False)
    # plot history

    pyplot.plot(history.history['loss'], label='train')
    #pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # pyplot.show()

    # make a prediction
    yhat = model.predict(test_X)
    # print("yhat:")
    # print(yhat)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)

    inv_y = inv_y[:, 0]
    end = time.time()
    # print('This took {} seconds.'.format(end - start))
    # calculate RMSE
    mse = mean_squared_error(inv_y, inv_yhat)
    rmse = sqrt(mse)

    def plot_predicted(predicted_data, true_data):
        fig, ax = plt.subplots(figsize=(17, 8))
        ax.set_title('Prediction vs. Actual after 100 epochs of training')
        ax.plot(true_data, label='True Data', color='green', linewidth='3')

        ax.plot(predicted_data, label='Prediction', color='red', linewidth='2')
        plt.legend()
        # plt.show()

    def plot_preds_actual(preds, actual, station_name, str_legend):
        text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig_verify = plt.figure(figsize=(17, 8))
        plt.plot(preds, color="blue")
        plt.plot(actual, color="orange")
        plt.title('AV pour LSTM:' + station_name)
        plt.ylabel('Pollution [PM10]')
        plt.xlabel('Daily timestep for 1 year')
        # str_legend = 'station: '+ str(station) + '\nR2 = ' + str(r2_score(test_y_expand, final_preds_expand)) + '\nMSE = ' + str(mse.item())
        # str_data = 'Train :' + begin_date + ' - ' + test_date  + '\nTest :' + begin_test + ' - ' + final_date
        first_legend = plt.legend(['predicted', 'actual data'], loc='upper left')
        dates_legend = plt.legend([str_legend], loc='upper right')
        ax = plt.gca().add_artist(first_legend)
        path_model_regression_verify = "./Seq2seq_juillet/" + folder + "/LSTM_lilleSeq2seq_lille" + text + ".png"
        fig_verify.savefig('./Data/_grid_' + station_name + "_" + text)
        #plt.show()

    str_legend = 'R2: ' + str(r2_score(inv_y, inv_yhat)) + "\nRMSE: " + str(rmse) + "\nMSE: " + str(mse) + "\nBatch=5, Lag=1"

    real_data = inv_y[-318:, ]

    np.savetxt("./Data/" + "/y_testLSTM.csv", real_data, delimiter=",")


    plot_preds_actual(inv_yhat[-318:, ], real_data, station, str_legend)


    #print('Root Mean Squared Error: {:.4f}'.format(rmse))

    # Calculate R^2 (regression score function)
    # print('Variance score: %.2f' % r2_score(y, data_pred))
    #print('Variance score: {:2f}'.format(r2_score(inv_y, inv_yhat)))

    #params = 'Batch: ' + str(BS) + ' #   Epochs: ' + str(ep) + ' #   R2: {:2f}'.format(r2_score(inv_y, inv_yhat)) + ' #   RMSE:' + str(rmse)

    params = 'TL: ' + str(tl) + ' #   R2: {:2f}'.format(r2_score(inv_y, inv_yhat)) + ' #   RMSE:' + str(rmse)

    print(params)

    write_results(params)
    #rmse_batch.append(rmse)
    #TimeLag.append(tl)
    RMSE.append(rmse)
    R2.append(r2_score(inv_y, inv_yhat))
    #Batches.append(BS)
datasets = {'salouel': salouel_raw_data, 'roth': roth_raw_data, 'creil': creil_raw_data}


#BSs = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]
#BSs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
BSs = [1, 3, 5, 7, 10, 15, 20,  30, 40, 50]
Epocs = [1, 2, 4, 6, 9, 10, 12, 15, 20, 24, 25, 29, 33, 37, 40, 45, 49, 50, 54, 59, 62, 66, 72, 75, 80, 85, 90, 99]
funcs = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'linear']
TimeLag = [1,2,4,6,8,10,12,15,20]

for tl in TimeLag:
    test_station(creil_raw_data, "creil", tl)



print(TimeLag)
#print(Batches)
print(R2)

print(RMSE)
"""
for bs in BSs:
    test_station(creil_raw_data, "creil", bs)


for f in funcs:
    test_station(creil_raw_data, "salouel", 5, 25, f)
    
    
for bs in BSs:
    test_station(creil_raw_data, "Creil", bs)


for bs in BSs:
    for ep in Epocs:
        test_station(creil_raw_data, "salouel", bs, ep)




#RMSEs = [9.86, 8.84, 8.94, 8.86, 9.09, 9.21, 9.67, 9.45, 9.81, 9.87]

for bs in BSs:
    test_station(salouel_raw_data, "salouel", bs)
#print(rmse_batch)
print(RMSE)
print(R2)


fig, ax = plt.subplots(figsize=(8,5))
ax.set_title('LSTM sea to scalar RMSE vs Batch Size salouel', fontsize=16)
ax.set_ylabel('RMSE', fontsize=16)
ax.set_xlabel('Batch size', fontsize=16)
ax.plot(BSs, rmse_batch, marker='x')
# plt.axvline(x=30, color='red', linestyle='dashed', linewidth=1)
plt.show()
fig.savefig('./SVR/batch_sal')

for key in datasets.keys():
    test_data(datasets[key], key)
    print("-----------------")



LAGS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

for LAG in LAGS:
    test_station(salouel_raw_data, "salouel", LAG)

print(RMSE)
print(R2)

#test_station(salouel_raw_data, "salouel")
"""

#test_station(salouel_raw_data, "salouel")

