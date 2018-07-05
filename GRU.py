import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import keras.backend as K
import keras
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error
from keras.layers.core import Activation
import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from math import sqrt
from sklearn.metrics import r2_score
from Column_settings import *
from W_settings import *
from pollution_plots import *

import matplotlib.patches as mpatches
from keras.layers import GRU

from sklearn.preprocessing import MinMaxScaler

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

    with open("results.txt", "a") as file:
        text = now.strftime("%Y-%m-%d %H:%M:%S") + ": " + "\n" + text + "\n" + "\n"
        file.write(text)


def scale_transformer(val, column_const):
    # print(column_const)
    return (val - column_const['min']) / (column_const['max'] - column_const['min'])


def reverse_transformer(val, column_const):
    return val * (column_const['max'] - column_const['min']) + column_const['min']


def reverse_y(val, col_PM10):
    return (val - col_PM10['min']) / (col_PM10['max'] - col_PM10['min'])


def r2_keras(y_true, y_pred):
    """Coefficient of Determination
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


salouel_raw_data = pd.read_csv("Moyennes_J_salouel_2005_2015.csv", header=None, sep=';', decimal=',')

creil_raw_data = pd.read_csv("Moyennes_J_creil_2005_2015.csv", header=None, sep=';', decimal=',')

cols = ['Date', 'NO2', 'O3', 'PM10', 'RR', 'TN', 'TX', 'TM',
        'PMERM', 'FFM', 'DXY', 'UM', 'GLOT']

cols_new = ['PM10', 'RR', 'TN', 'TX', 'TM',
        'PMERM', 'FFM', 'DXY', 'UM', 'GLOT']


roth_raw_data = pd.read_csv("Moyennes_J_roth_2005_2015.csv", header=None, sep=';', decimal=',')

arr_row_data = [salouel_raw_data, creil_raw_data, roth_raw_data]

"""
Deleting dates column and making it all floats
"""


def preprocessing_data(df):
    df = df.iloc[1:]
    df.columns = cols
    df = df.drop(["Date", 'NO2', 'O3'], axis=1)
    df = df.astype(float)
    return df


"""
TODO:
def prepare_data_ar(array_of_df)
"""

salouel_data = preprocessing_data(salouel_raw_data)
creil_data = preprocessing_data(creil_raw_data)
roth_data = preprocessing_data(roth_raw_data)

"""
DELETE ROWS WITH MISSING VALUES
"""

creil_data = creil_data.dropna(axis=0, how="any")
salouel_data = salouel_data.dropna(axis=0, how="any")
roth_data = roth_data.dropna(axis=0, how="any")

merged_data = pd.concat([salouel_data, creil_data, roth_data])

# Sorting by date, returns TrainModel, TestModel and Control dataframes


# REPLACE MISSING VALUES BY AVERAGE OF CORRESPONDING VALUES


"""
Cut data until test_date

Returen Y and X

"""
test_date = '31/12/2014'

"""
NORMALISATION
"""


# salouel_data["PM10"] = salouel_data["PM10"].apply(reverse_transformer, args=(settings_array["PM10"],))


def get_season(dt):
    # it returnts string
    # print('type of dt:' + str(type(dt)))
    # print(dt)
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


separation_date = '31/12/2010'
begin_test = '01/01/2015'
test_date = '31/12/2014'
begin_date = '01/01/2005'
final_date = '31/12/2015'


def prepare_data(df_temp, keep_season=False, keep_wind=False):
    train_data = df_temp[df_temp['Date'] < test_date].drop(["Date"], axis=1).dropna(axis=0, how="any")
    test_data = df_temp[df_temp['Date'] > test_date].drop(["Date"], axis=1).dropna(axis=0, how="any")

    season_train = train_data['Season']
    season_test = test_data['Season']

    train_data = train_data.drop("Season", axis=1)
    test_data = test_data.drop("Season", axis=1)

    vent_train = train_data["vent"]
    vent_test = test_data["vent"]

    train_data = train_data.drop("vent", axis=1)
    test_data = test_data.drop("vent", axis=1)

    """

    if (keep_season):
        train_data["Season"] = season_train
        test_data["Season"] = season_test

    if (keep_wind):
        train_data["vent"] = vent_train
        test_data["vent"] = vent_test

    train_data = pd.get_dummies(train_data)
    test_data = pd.get_dummies(test_data)
    """
    train_data.to_csv('./data/train.csv')

    test_data.to_csv('./data/test.csv')

    print("Dataframes saved...")
    return train_data, test_data


def clean_data(array_raw_data, cut):
    global begin_date
    # concat is first
    conc = True
    df = array_raw_data.iloc[1:]
    df.columns = cols
    df_temp = df.drop(["Date", 'NO2', 'O3'], axis=1)
    df_temp = df_temp.astype(float)
    df_temp["Date"] = pd.to_datetime(df["Date"])
    if cut == True:
        df_temp = df_temp[df_temp['Date'] > separation_date]
        begin_date = '01/01/2010'
    else:
        begin_date = '01/01/2005'
    # df_temp["Date"] = pd.to_datetime(df["Date"])
    df_temp['Season'] = df_temp["Date"].apply(get_season)
    df_temp['vent'] = df_temp['DXY'].apply(get_dir)

    return df_temp


n_past = 90
n_future = 0


def gen_sequence(df):
    X_seq = []
    for i in range(n_past, len(df) - n_future + 1):
        X_seq.append(df.iloc[i - n_past: i, :].values)
    return X_seq


def test_station(data, station, cut):
    text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scaler = MinMaxScaler(feature_range=(0,1))
    scalerTest = MinMaxScaler(feature_range=(0,1))
    scalerTrain = MinMaxScaler(feature_range=(0,1))

    df_temp = clean_data(data, cut).dropna(axis=0, how="any")
    print(df_temp.head())

    # Pollution plots go here

    #display_plot(df_temp)
    df_date = df_temp['Date']

    df_temp = df_temp.drop(columns=["Season", "vent", 'Date'])
    #df_temp = df_temp.drop(["vent"])
    print(df_temp.head())

    #df_temp = df_temp.drop(['Date'])

    #df_temp = scaler.fit_transform(df_temp)
    df_temp = pd.DataFrame(df_temp, columns=cols_new)
    df_temp['Date'] = df_date

    print(df_temp.head())

    train_data = df_temp[df_temp['Date'] < test_date].drop(["Date"], axis=1).dropna(axis=0, how="any")
    test_data = df_temp[df_temp['Date'] > test_date].drop(["Date"], axis=1).dropna(axis=0, how="any")
    train_data = pd.DataFrame(scalerTrain.fit_transform(train_data), columns=cols_new)
    test_data = pd.DataFrame(scalerTest.fit_transform(test_data), columns=cols_new)


    """
    season_train = train_data['Season']
    season_test = test_data['Season']

    train_data = train_data.drop("Season", axis=1)
    test_data = test_data.drop("Season", axis=1)

    vent_train = train_data["vent"]
    vent_test = test_data["vent"]

    train_data = train_data.drop("vent", axis=1)
    test_data = test_data.drop("vent", axis=1)

    
    train_data, test_data = prepare_data(df_temp, True, False)
    print(test_data.head())

    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)

    train_data = pd.DataFrame(train_data, columns=cols_new)
    test_data = pd.DataFrame(test_data, columns=cols_new)
    print(test_data.head())
    """

    train_y_merged_scaled_data = train_data['PM10']
    train_x_merged_scaled_data = train_data.drop("PM10", axis=1)
    test_y_merged_scaled_data = test_data['PM10']
    test_x_merged_scaled_data = test_data.drop("PM10", axis=1)

    col_string = " ".join(str(x) for x in train_x_merged_scaled_data.columns)

    print(test_x_merged_scaled_data.info())
    print("y length")
    print(test_y_merged_scaled_data.shape)

    train_seq_merged_df = np.array(gen_sequence(train_x_merged_scaled_data))
    test_seq_merged_df = np.array(gen_sequence(test_x_merged_scaled_data))

    # y_scaled_data

    train_y_merged_scaled_data = train_y_merged_scaled_data.values.reshape(len(train_y_merged_scaled_data),
                                                                           1)  # [ [] [] []]
    test_y_merged_scaled_data = test_y_merged_scaled_data.values.reshape(len(test_y_merged_scaled_data), 1)

    nb_features = train_seq_merged_df.shape[2]

    nb_features_merged = train_seq_merged_df.shape[2]
    """
    # modéle 1
    model = Sequential()
    model.add(LSTM(
             input_shape=(n_past, nb_features),
             units=100,
             return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
              units=50,
              return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', r2_keras])


    # Modéle 2 
    model = Sequential()  # <- this model works better
    model.add(LSTM(50, input_shape=(n_past, nb_features)))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    """
    model = Sequential()
    model.add(GRU(units=100, activation=None, input_shape=(n_past, nb_features), return_sequences=False))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('relu'))
    # model.load_weights('weights/bitcoin2015to2017_close_GRU_1_tanh_relu_-32-0.00004.hdf5')
    model.compile(loss='mse', optimizer='adam')

    print(model.summary())

    # model_path = './regression_model.h5'

    model_path_merged = './regression_model_merged.h5'
    print("Longuer:")
    print(len(train_y_merged_scaled_data))
    train_merged_label_array = train_y_merged_scaled_data[n_past + n_future - 1:len(
        train_y_merged_scaled_data)]
    test_merged_y_scaled_data = test_y_merged_scaled_data[n_past + n_future - 1:len(
        test_y_merged_scaled_data)]

    EPOCHS = 2
    BATCH_SIZE = 200

    print("shape of testX :")
    print(test_seq_merged_df.shape)

    print("shape of testY :")
    print(test_merged_y_scaled_data.shape)

    print("shape of trainX :")
    print(train_seq_merged_df.shape)

    print("shape of trainY :")
    print(train_merged_label_array.shape)

    history = model.fit(train_seq_merged_df, train_merged_label_array, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(test_seq_merged_df, test_merged_y_scaled_data), verbose=2,
                        )

    # estimator = load_model(model_path_merged, custom_objects={'r2_keras': r2_keras})

    index = (next(i for i, j in enumerate(arr_row_data) if j is roth_raw_data) + 1) % len(
        arr_row_data)  # gives index of the next element

    dt = pd.read_csv("concatenated_data.csv", sep=',', decimal=',')

    # X_test,Y_test = gen_test_data1(arr_row_data[index])
    # X_test, Y_test = gen_test_data1(dt)

    print(train_seq_merged_df.shape)
    print(train_merged_label_array.shape)
    # print(np.array(X_test.shape))
    # print(Y_test.shape)

    # scores_test = estimator.evaluate(X_test, Y_test, verbose=2) # Here is a problem

    # print('\nR^2: {}'.format(scores_test[2]))
    # print('\nMAE: {}'.format(scores_test[1]))
    # print('\nR^2: {}'.format(scores_test[2]))

    # final_preds = estimator.predict(X_test)
    # print(type(final_preds))
    print(type(test_seq_merged_df))
    print(test_seq_merged_df.shape)



    final_preds = model.predict(test_seq_merged_df)
    predicted_values = pd.DataFrame(final_preds)
    print(test_x_merged_scaled_data.head())
    final_df = test_x_merged_scaled_data.loc[89:, :]
    print(final_preds.shape)
    print(final_df.info())

    final_df['PM10'] = predicted_values
    print(final_df.head())
    print(final_df.info())

    print(type(final_preds))
## MERGE IT
    print(predicted_values)

    predicted_reversed = scalerTest.inverse_transform(final_df)



    Y_reversed = scaler.inverse_transform(pd.DataFrame(test_merged_y_scaled_data))



    mse = np.mean((predicted_reversed - Y_reversed) ** 2)

    # rmse = sqrt(mean_squared_error(Y_reversed, predicted_reversed))

    print('Test MSE: ' + str(mse))
    print('Variance score: {:2f}'.format(r2_score(Y_reversed, predicted_reversed)))

    fig_verify = plt.figure()
    plt.plot(predicted_reversed, color="blue")
    plt.plot(Y_reversed, color="orange")
    plt.title('LSTM2 Train :' + begin_date + ' - ' + test_date + '\nTest :' + begin_test + ' - ' + final_date)
    plt.ylabel('value')
    plt.xlabel('row')
    str_legend = 'station: ' + str(station) + '\nR2 = ' + str(
        r2_score(Y_reversed, predicted_reversed)) + '\nMSE = ' + str(mse.item())
    # str_data = 'Train :' + begin_date + ' - ' + test_date  + '\nTest :' + begin_test + ' - ' + final_date
    first_legend = plt.legend(['predicted', 'actual data'], loc='upper left')
    dates_legend = plt.legend([str_legend], loc='upper right')

    # mesures_legend = plt.legend([str_legend], loc='upper right')

    ax = plt.gca().add_artist(first_legend)

    # ay = plt.gca().add_artist(mesures_legend)

    # plt.legend([AnyObject()], [str_legend],
    #          handler_map={AnyObject: AnyObjectHandler()})

    # plt.show()
    path_model_regression_verify = "./LSTM2_lille" + text + ".png"

    fig_verify.savefig(path_model_regression_verify)

    write_results(str_legend)
    # os.remove(model_path_merged)


# test_station(salouel_raw_data, "Sal", True) #  37.523128 0.255424 <- 1er modele ! 2eme modele: R2 0.307077 ! MSE :  33.647119

dict = {'salouel': salouel_raw_data, 'roth': roth_raw_data, 'creil': creil_raw_data}

# Test plot pollution#
# test_station(creil_raw_data, "Creil", True)


for key in dict.keys():
    test_station(dict[key], key, False)
    test_station(dict[key], key, True)
"""

for st in arr_row_data: # the best result is 2 -> saluel True
    test_station(st, False)
    test_station(st, True)

#dt = pd.read_csv("concatenated_data.csv", sep=',', decimal=',')
test_station(creil_raw_data, "creil", True)
"""