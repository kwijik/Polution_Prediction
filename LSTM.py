from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
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
from matplotlib import pyplot as plt

import datetime
import os

folder = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

cols = ['Date', 'NO2', 'O3', 'PM10', 'RR', 'TN', 'TX', 'TM',
                     'PMERM', 'FFM', 'DXY', 'UM', 'GLOT']

rmse_batch = []

def preprocessing_data(df):
    df = df.iloc[1:]
    df.columns = cols
    df = df.drop(["Date", 'NO2', 'O3'], axis=1)
    df = df.astype(float)
    df['vent'] = df['DXY'].apply(get_dir)
    df = df.drop(['DXY'], axis=1)


    #df = df.dropna(axis=0, how="any")

    return df

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
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
    #print("cols:")
    #print(cols)
    agg = concat(cols, axis=1)

    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def write_results(text):
    # file = open("results.txt", "w")

    now = datetime.datetime.now()

    with open("batch.txt", "a") as file:
        text = now.strftime("%Y-%m-%d %H:%M:%S") + ": " + "\n" + text + "\n" + "\n"
        file.write(text)


creil_raw_data = pd.read_csv("Moyennes_J_creil_2005_2015.csv", header=None, sep=';', decimal=',')
roth_raw_data = pd.read_csv("Moyennes_J_roth_2005_2015.csv", header=None, sep=';', decimal=',')
salouel_raw_data = pd.read_csv("Moyennes_J_salouel_2005_2015.csv", header=None, sep=';', decimal=',')

#print(data)
def test_data(dataset, station_name, BS):
    global rmse_batch
    params = "Test for " + station_name + "; Batch_size is: " + str(BS)
    print(params)

    dataset = preprocessing_data(dataset)
    values = dataset.values

    encoder = LabelEncoder()
    values[:,9] = encoder.fit_transform(values[:,9])

    reframed = series_to_supervised(values)



    reframed.drop(reframed.columns[[-9, -8, -7,-6,-5,-4,-3,-2,-1]], axis=1, inplace=True)

    #print("####")
    #print(reframed.head())
    #print("####")

    values = reframed.values
    #print("values:")
    #print(values.shape)

    ########
    # LSTM #
    ########

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(values[:,:-1])
    scaled_label = scaler.fit_transform(values[:,-1].reshape(-1,1))
    values = np.column_stack((scaled_features, scaled_label))

    #print("values:")
    #print(values.shape)

    n_train_hours = 365
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    #print("test:")
    #print(test.shape)
    # split into input and outputs
    # features take all values except the var1
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    #print("train_X:")
    #print(test_X)

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    #print("train_X:")
    #print(test_X.shape)
    #print(test_X.info())

    #print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # design network
    model = Sequential()
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')


    #print(reframed.head())

    start = time.time()

    # fit network
    ###################### Can change Epochs, Batch size here #######################
    history = model.fit(train_X, train_y, epochs=25, batch_size=BS, validation_data=(test_X, test_y),
                        verbose=1, shuffle=False)
    # plot history

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    #pyplot.show()



    # make a prediction
    yhat = model.predict(test_X)
    #print("yhat:")
    #print(yhat)
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
    #print('This took {} seconds.'.format(end - start))
    # calculate RMSE
    mse = mean_squared_error(inv_y, inv_yhat)
    rmse = sqrt(mse)
    print('Test RMSE: %.3f' % rmse)




    def plot_predicted(predicted_data, true_data):
        fig, ax = plt.subplots(figsize=(17,8))
        ax.set_title('Prediction vs. Actual after 100 epochs of training')
        ax.plot(true_data, label='True Data', color='green', linewidth='3')

        ax.plot(predicted_data, label='Prediction', color='red', linewidth='2')
        plt.legend()
        #plt.show()


    def plot_preds_actual(preds, actual, station_name, str_legend):
        text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig_verify = plt.figure()
        plt.plot(preds, color="blue")
        plt.plot(actual, color="orange")
        plt.title('LSTM Train :' + station_name)
        plt.ylabel('value')
        plt.xlabel('row')
        # str_legend = 'station: '+ str(station) + '\nR2 = ' + str(r2_score(test_y_expand, final_preds_expand)) + '\nMSE = ' + str(mse.item())
        # str_data = 'Train :' + begin_date + ' - ' + test_date  + '\nTest :' + begin_test + ' - ' + final_date
        first_legend = plt.legend(['predicted', 'actual data'], loc='upper left')
        dates_legend = plt.legend([str_legend], loc='upper right')
        ax = plt.gca().add_artist(first_legend)
        path_model_regression_verify = "./Seq2seq_juillet/" + folder + "/Seq2seq_lilleSeq2seq_lille" + text + ".png"
        fig_verify.savefig('./SVR/' +station_name + "_" + text)


    str_legend = 'R2: ' + str(r2_score(inv_y, inv_yhat)) + "\nRMSE: " + str(rmse) + "\nMSE: " + str(mse)

    plot_preds_actual(inv_yhat[:300,], inv_y[:300,], station_name, str_legend)

    print('Root Mean Squared Error: {:.4f}'.format(rmse))

    #Calculate R^2 (regression score function)
    #print('Variance score: %.2f' % r2_score(y, data_pred))
    print('Variance score: {:2f}'.format(r2_score(inv_y, inv_yhat)))

    write_results(params + "; RMSE " + str(rmse)  + "; R2 " + str(r2_score(inv_y, inv_yhat)))
    rmse_batch.append(rmse)

datasets =  {'salouel': salouel_raw_data, 'roth': roth_raw_data, 'creil': creil_raw_data}

BSs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
RMSEs = [9.86, 8.84, 8.94, 8.86, 9.09, 9.21, 9.67, 9.45, 9.81, 9.87]




for bs in BSs:
    test_data(salouel_raw_data, "salouel", bs)
print(rmse_batch)


fig, ax = plt.subplots(figsize=(8,5))
ax.set_title('LSTM sea to scalar RMSE vs Batch Size salouel', fontsize=16)
ax.set_ylabel('RMSE', fontsize=16)
ax.set_xlabel('Batch size', fontsize=16)
ax.plot(BSs, rmse_batch, marker='x')
# plt.axvline(x=30, color='red', linestyle='dashed', linewidth=1)
plt.show()
fig.savefig('./SVR/batch_sal')

"""
for key in datasets.keys():
    test_data(datasets[key], key)
"""