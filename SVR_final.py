import pandas as pd
import numpy as np
import datetime
import os, sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pollution_plots import *
from sklearn.svm import SVR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(7)


folder = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

batch_sizes = []
times_test = []
times_train = []



# save results in file

def write_results(text):
    # file = open("results.txt", "w")

    now = datetime.datetime.now()

    with open("svr_grid.txt", "a") as file:
        text = now.strftime("%Y-%m-%d %H:%M:%S") + ": " + "\n" + text + "\n" + "\n"
        file.write(text)

# reading data from files
salouel_raw_data = pd.read_csv("./propositio/salou.csv", header=None, sep=';', decimal=',')
creil_raw_data = pd.read_csv("./propositio/creil.csv", header=None, sep=';', decimal=',')
roth_raw_data = pd.read_csv("./propositio/roth.csv", header=None, sep=';', decimal=',')

cols = ['Date', 'PM10', 'RR', 'TN', 'TX', 'TM',
                     'PMERM', 'FFM', 'UM', 'GLOT', 'DXY']


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
    print('in series to suprsd')
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



# main function
# input: data - dataframe, station - string
# output: void

def test_station(data, station):

    dataset = preprocessing_data(data)
    print(dataset.head())

    #Traitement de données
    dataset = dataset.drop(["TX", 'TN', 'DXY'], axis=1)

    values = dataset.values

    values = pd.DataFrame(values).dropna().values

    reframed = series_to_supervised(values, n_in=1)
    reframed.drop(reframed.columns[[-6, -5, -4, -3, -2, -1]], axis=1, inplace=True)


    values = reframed.values
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    scaled_features = scaler.fit_transform(values[:,:])

    scaled_label = scaler.fit_transform(values[:,-1].reshape(-1,1))
    values = np.column_stack((scaled_features, scaled_label))

    n_train_days = 359 * 3
    test_days = n_train_days + 365
    train = values[:n_train_days, :]
    test = values[test_days:, :]
    validation = values[n_train_days:test_days, :]

    # split into input and outputs
    # features take all values except the var1
    validation_X, validation_Y = validation[:, :-1], validation[:, -1]
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]))
    validation_X = validation_X.reshape((validation_X.shape[0], validation_X.shape[1]))

    # design network
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

    y_pred = scaler.inverse_transform(data_pred.reshape(-1,1))
    y_inv = scaler.inverse_transform(test_y.reshape(-1,1))


    mse = mean_squared_error(y_inv, y_pred)
    rmse = np.sqrt(mse)
    print('Mean Squared Error: {:.4f}'.format(mse))
    print('Root Mean Squared Error: {:.4f}'.format(rmse))
    print('Variance score: {:2f}'.format(r2_score(y_inv, y_pred)))

    # plot function
    # input: preds - np array, actual - np array, station_name - strng, str_legend - string
    # output: creates graph with pred/real results
    def plot_preds_actual(preds, actual, station_name, str_legend):
        text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig_verify = plt.figure(figsize=(17,8))
        plt.plot(preds, color="blue")
        plt.plot(actual, color="orange")
        plt.title('SVR: ' + station_name + ' SV test')
        plt.ylabel('value')
        plt.xlabel('row')
        first_legend = plt.legend(['predicted', 'actual data'], loc='upper left')
        dates_legend = plt.legend([str_legend], loc='upper right')
        ax = plt.gca().add_artist(first_legend)
        path_model_regression_verify = "./Seq2seq_juillet/" + folder + "/Seq2seq_lilleSeq2seq_lille" + text + ".png"
        plt.show()
        fig_verify.savefig('./Images/svr__' +station_name )

    str_legend = 'R2: ' + str(r2_score(y_inv, y_pred)) + "\nRMSE: " + str(rmse) + "\nMSE: " + str(mse) + "\nGamma:0.7; C:0.9"
    divider = - 329
    y_real = y_inv[divider:,]
    plot_preds_actual(y_pred[divider:,], y_inv[divider:,], station, str_legend)
    res = "Gamma: " + str(0.7) + " #  C: " + str(0.9) +  " #  RMSE: " + str(rmse)  + " #  R2: " + str(r2_score(y_inv, y_pred))
    write_results(res)


test_station(roth_raw_data, "Roth")

