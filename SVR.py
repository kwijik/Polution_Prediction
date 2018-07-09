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


cols = ['Date', 'NO2', 'O3', 'PM10', 'RR', 'TN', 'TX', 'TM',
                     'PMERM', 'FFM', 'DXY', 'UM', 'GLOT']


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



from pandas import read_csv
dataset = pd.read_csv("Moyennes_J_salouel_2005_2015.csv", header=None, sep=';', decimal=',')
#print(data)

def preprocessing_data(df):
    df = df.iloc[1:]
    df.columns = cols
    df = df.drop(["Date", 'NO2', 'O3'], axis=1)
    df = df.astype(float)
    df['vent'] = df['DXY'].apply(get_dir)
    df = df.drop(['DXY'], axis=1)


    #df = df.dropna(axis=0, how="any")

    return df

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


def plot_preds_actual(preds, actual):
    fig, ax = plt.subplots(figsize=(17,8))
    ax.plot(preds, color='red', label='Predicted data')
    ax.plot(actual, color='green', label='True data')
    ax.set_xlabel('Hourly Timestep in First Month of Predicted Year', fontsize=16)
    ax.set_ylabel('Pollution [pm2.5]', fontsize=16)
    ax.set_title('Nonlinear Regression using SVR on Test set', fontsize=16)
    ax.legend()
    plt.show()

plot_preds_actual(y_pred[:24*31*1,], y_inv[:24*31*1,])

