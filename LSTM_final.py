import pandas as pd
import numpy as np
import datetime

import time
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pollution_plots import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(7)


folder = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")



# changed all , by . in cvs files

def write_results(text):
    # file = open("results.txt", "w")

    now = datetime.datetime.now()

    with open("grid.txt", "a") as file:
        text = now.strftime("%Y-%m-%d %H:%M:%S") + ": " + "\n" + text + "\n" + "\n"
        file.write(text)


# reading data from files
salouel_raw_data = pd.read_csv("./propositio/salou.csv", header=None, sep=';', decimal=',')
roth_raw_data = pd.read_csv("./propositio/roth.csv", header=None, sep=';', decimal=',')
creil_raw_data = pd.read_csv("./propositio/creil.csv", header=None, sep=';', decimal=',')

cols = ['Date', 'PM10', 'RR', 'TN', 'TX', 'TM',
                     'PMERM', 'FFM', 'UM', 'GLOT', 'DXY']


arr_row_data = [salouel_raw_data, creil_raw_data, roth_raw_data]


# Deleting dates column and making it all floats
# input df (dataframe), output df (dataframe)
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



#separation_date = '31/12/2010'
begin_test = '01/01/2015'
test_date = '31/12/2013'
begin_date = '01/01/2010'
#final_date = '31/12/2015'
validation_date = '31/12/2014'

# adds lagged columns
# input: data - dataframe, n_in - int, n_out - int, dropnan - boolean
# output: dataframe
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    #print('in series to suprsd')
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    if dropnan:
        df.dropna(inplace=True)
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
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values

    if dropnan:
        agg.dropna(inplace = True)

    return agg

n_past = 90
n_future = 10


# main function
# input: data - dataframe, station - string
# output: void
def test_station(data, station):

    # Traitement de data
    dataset = preprocessing_data(data)
    dataset = dataset.drop([ 'TN'], axis=1)

    values = dataset.values
    values = pd.DataFrame(values).dropna().values
    encoder = LabelEncoder()
    values[:, 8] = encoder.fit_transform(values[:, 8])


    reframed = series_to_supervised(values, n_in=1)
    reframed.drop(reframed.columns[[-8, -7,-6, -5, -4, -3, -2, -1]], axis=1, inplace=True)


    values = reframed.values

    ########
    # LSTM #
    ########

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(values[:, :-1])
    scaled_label = scaler.fit_transform(values[:, -1].reshape(-1, 1))
    values = np.column_stack((scaled_features, scaled_label))

    observed_days = 359
    days_in_years = 365
    n_train_days = observed_days * 3
    test_days = n_train_days+days_in_years
    train = values[:n_train_days, :]
    test = values[test_days:, :]
    validation = values[n_train_days:test_days, :]
    # split into input and outputs
    # features take all values except the var1
    validation_X, validation_Y = validation[:, :-1], validation[:, -1]
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]


    # reshape input to be 3D [samples, timesteps, features]
    validation_X = validation_X.reshape((validation_X.shape[0], 1, validation_X.shape[1]))
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # print(test_X.info())

    # design network
    LSTM_neurones = 128
    Dense_neurones = 50

    model = Sequential()
    model.add(LSTM(LSTM_neurones, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(Dense_neurones, activation='softmax'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # print(reframed.head())

    number_epochs = 99
    batch_size = 20

    # fit network
    ###################### Can change Epochs, Batch size here #######################
    history = model.fit(train_X, train_y, epochs=number_epochs, batch_size=batch_size, validation_data=(validation_X, validation_Y),
                        verbose=1, shuffle=False)

    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.legend()


    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)

    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    end = time.time()
    # calculate RMSE
    mse = mean_squared_error(inv_y, inv_yhat)
    rmse = sqrt(mse)

    def plot_predicted(predicted_data, true_data):
        fig, ax = plt.subplots(figsize=(17, 8))
        ax.set_title('Prediction vs. Actual after 100 epochs of training')
        ax.plot(true_data, label='True Data', color='green', linewidth='3')
        ax.plot(predicted_data, label='Prediction', color='red', linewidth='2')
        plt.legend()

    def plot_preds_actual(preds, actual, station_name, str_legend):
        text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig_verify = plt.figure(figsize=(17, 8))
        plt.plot(preds, color="blue")
        plt.plot(actual, color="orange")
        plt.title('SV pour LSTM:' + station_name + " test")
        plt.ylabel('Pollution [PM10]')
        plt.xlabel('Daily timestep for 1 year')
        first_legend = plt.legend(['predicted', 'actual data'], loc='upper left')
        dates_legend = plt.legend([str_legend], loc='upper right')
        ax = plt.gca().add_artist(first_legend)
        path_model_regression_verify = "./Seq2seq_juillet/" + folder + "/LSTM_lilleSeq2seq_lille" + text + ".png"
        fig_verify.savefig('./Images/' + station_name + "_" + text)
        plt.show()

    days_to_show = 318
    str_legend = 'R2: ' + str(r2_score(inv_y, inv_yhat)) + "\nRMSE: " + str(rmse) + "\nMSE: " + str(mse) + "\nBatch=20, Lag=1"
    real_data = inv_y[-days_to_show:, ]
    plot_preds_actual(inv_yhat[-days_to_show:, ], real_data, station, str_legend)
    # Calculate R^2 (regression score function)

    params = 'Batch: ' + str(batch_size) + ' #   Epochs: ' + str(number_epochs) + ' #   R2: {:2f}'.format(r2_score(inv_y, inv_yhat)) + ' #   RMSE:' + str(rmse)
    print(params)

    write_results(params)


test_station(roth_raw_data, "Roth")
