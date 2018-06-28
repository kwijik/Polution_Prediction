
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import keras.backend as K
import keras
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error
from keras.layers.core import Activation
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from math import sqrt
from sklearn.metrics import r2_score
from Column_settings import *


import matplotlib.patches as mpatches

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

#changed all , by . in cvs files

def write_results(text):
    #file = open("results.txt", "w")

    now = datetime.datetime.now()

    with open("results.txt", "a") as file:
        text = now.strftime("%Y-%m-%d %H:%M:%S") + ": " + "\n" + text + "\n" + "\n"
        file.write(text)

def scale_transformer(val, column_const):
    #print(column_const)
    return (val - column_const['min']) / (column_const['max'] - column_const['min'])

def reverse_transformer(val, column_const):
    return val * (column_const['max'] - column_const['min']) + column_const['min']

def reverse_y(val, col_PM10):
    return (val - col_PM10['min']) / (col_PM10['max'] - col_PM10['min'])


def r2_keras(y_true, y_pred):
    """Coefficient of Determination
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()))


salouel_raw_data = pd.read_csv("Moyennes_J_salouel_2005_2015.csv", header=None, sep=';', decimal=',')


creil_raw_data = pd.read_csv("Moyennes_J_creil_2005_2015.csv", header=None, sep=';', decimal=',')

cols = ['Date', 'NO2', 'O3', 'PM10', 'RR', 'TN', 'TX', 'TM',
                     'PMERM', 'FFM', 'DXY', 'UM', 'GLOT']

roth_raw_data = pd.read_csv("Moyennes_J_roth_2005_2015.csv", header=None, sep=';', decimal=',')


arr_row_data = [salouel_raw_data, creil_raw_data, roth_raw_data]


"""
Deleting dates column and making it all floats
"""

def prepare_data(df):
    df = df.iloc[1:]
    df.columns = cols
    df = df.drop(["Date", 'NO2', 'O3'], axis=1)
    df = df.astype(float)
    return df

"""
TODO:
def prepare_data_ar(array_of_df)
"""

salouel_data = prepare_data(salouel_raw_data)
creil_data = prepare_data(creil_raw_data)
roth_data = prepare_data(roth_raw_data)

"""
DELETE ROWS WITH MISSING VALUES
"""


creil_data = creil_data.dropna(axis=0, how="any")
salouel_data = salouel_data.dropna(axis=0, how="any")
roth_data = roth_data.dropna(axis=0, how="any")

merged_data = pd.concat([salouel_data, creil_data, roth_data])

# Sorting by date, returns TrainModel, TestModel and Control dataframes


#REPLACE MISSING VALUES BY AVERAGE OF CORRESPONDING VALUES



"""
Cut data until test_date

Returen Y and X

"""
test_date = '31/12/2014'

def gen_test_data(array_raw_data):
    # concat is first
    df = array_raw_data.iloc[1:]
    df.columns = cols
    df_temp = df.drop(["Date", 'NO2', 'O3'], axis=1)
    df_temp = df_temp.astype(float)
    df_temp["Date"] = pd.to_datetime(df["Date"])

    df_temp = df_temp[df_temp['Date'] > test_date].dropna(axis=0, how="any")

    Y = df_temp['PM10']
    X = df_temp[['RR', 'TN', 'TX', 'TM',
                     'PMERM', 'FFM', 'DXY', 'UM', 'GLOT']]
    return Y, X


def gen_test_data1(array_raw_data): # Fixed transformation into sequences!!!
    # concat is first
    df = array_raw_data.iloc[1:]
    df.columns = cols
    df_temp = df.drop(["Date", 'NO2', 'O3'], axis=1)
    df_temp = df_temp.astype(float)
    for col in df_temp.columns:
        df_temp[col] = df_temp[col].apply(scale_transformer, args=(settings_array[col],))
    df_temp["Date"] = pd.to_datetime(df["Date"])

    df_temp = df_temp[df_temp['Date'] > test_date].dropna(axis=0, how="any")



    Y = df_temp['PM10']
    X = df_temp[['RR', 'TN', 'TX', 'TM',
                     'PMERM', 'FFM', 'DXY', 'UM', 'GLOT']]
    X_seq = np.array(gen_sequence(X))


    X_np = np.array(X_seq)
    Y_reshaped = Y.values.reshape(len(Y), 1)

    Y_reshaped_resized = Y_reshaped[n_past + n_future - 1:len(Y_reshaped)]
    return X_np, Y_reshaped_resized

"""
NORMALISATION
"""

for col in salouel_data.columns:
    salouel_data[col] = salouel_data[col].apply(scale_transformer, args=(settings_array[col],))



#salouel_data["PM10"] = salouel_data["PM10"].apply(reverse_transformer, args=(settings_array["PM10"],))

separation_date = '31/12/2009'
begin_test = '01/01/2015'
test_date = '31/12/2014'
begin_date = '01/01/2005'
final_date = '31/12/2015'

def separate_data(array_raw_data, cut):
    global begin_date
    # concat is first
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
    #df_temp["Date"] = pd.to_datetime(df["Date"])
    train = df_temp[df_temp['Date'] < test_date].drop("Date", axis=1).dropna(axis=0, how="any")
    test = df_temp[df_temp['Date'] > test_date].drop("Date", axis=1).dropna(axis=0, how="any")
    return train, test


n_past = 90
n_future = 0

def gen_sequence(df):
    X_seq = []
    for i in range(n_past, len(df) - n_future + 1):
        X_seq.append(df.iloc[i - n_past : i, :].values)
    return X_seq


def test_station(data, station, cut):
    text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    train_data, test_data = separate_data(data, cut)

    for col in salouel_data.columns:
        train_data[col] = train_data[col].apply(scale_transformer, args=(settings_array[col],))

    for col in test_data.columns:
        test_data[col] = test_data[col].apply(scale_transformer, args=(settings_array[col],))

    train_y_merged_scaled_data = train_data['PM10']
    train_x_merged_scaled_data = train_data.drop("PM10", axis=1)
    test_y_merged_scaled_data = test_data['PM10']
    test_x_merged_scaled_data = test_data.drop("PM10", axis=1)

    col_string = " ".join(str(x) for x in train_x_merged_scaled_data.columns)


    train_seq_merged_df = np.array(gen_sequence(train_x_merged_scaled_data))
    test_seq_merged_df = np.array(gen_sequence(test_x_merged_scaled_data))



    #y_scaled_data

    train_y_merged_scaled_data = train_y_merged_scaled_data.values.reshape(len(train_y_merged_scaled_data), 1) # [ [] [] []]
    test_y_merged_scaled_data = test_y_merged_scaled_data.values.reshape(len(test_y_merged_scaled_data), 1)



    nb_features = train_seq_merged_df.shape[2]

    nb_features_merged = train_seq_merged_df.shape[2]


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
    """
    model = Sequential()  # <- this model works better
    model.add(LSTM(50, input_shape=(n_past, nb_features)))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
"""
    print(model.summary())


    #model_path = './regression_model.h5'

    model_path_merged = './regression_model_merged.h5'


    train_merged_label_array = train_y_merged_scaled_data[n_past + n_future - 1:len(train_y_merged_scaled_data)] # Ð¿Ñ€Ð¾Ð²ÐµÑ€Ð¸Ñ‚ÑŒ Ñ€Ð°Ð·Ð¼ÐµÑ€ (+-1)!
    test_merged_y_scaled_data = test_y_merged_scaled_data[n_past + n_future - 1:len(test_y_merged_scaled_data)] # Ð¿Ñ€Ð¾Ð²ÐµÑ€Ð¸Ñ‚ÑŒ Ñ€Ð°Ð·Ð¼ÐµÑ€ (+-1)!


    EPOCHS = 100
    BATCH_SIZE = 200

    print("shape of testX :")
    print(test_seq_merged_df.shape)

    print("shape of testY :")
    print(test_merged_y_scaled_data.shape)

    print("shape of trainX :")
    print(train_seq_merged_df.shape)

    print("shape of trainY :")
    print(train_merged_label_array.shape)

    history = model.fit(train_seq_merged_df, train_merged_label_array, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_seq_merged_df, test_merged_y_scaled_data), verbose=2,
              callbacks = [#keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                           keras.callbacks.ModelCheckpoint(model_path_merged, monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
              )

    estimator = load_model(model_path_merged, custom_objects={'r2_keras': r2_keras})

    index = (next(i for i, j in enumerate(arr_row_data) if j is roth_raw_data) + 1) % len(arr_row_data)  # gives index of the next element

    X_test,Y_test = gen_test_data1(arr_row_data[index])



    print(train_seq_merged_df.shape)
    print(train_merged_label_array.shape)
    print(np.array(X_test.shape))
    print(Y_test.shape)



    scores_test = estimator.evaluate(X_test, Y_test, verbose=2) # Here is a problem


    #print('\nR^2: {}'.format(scores_test[2]))
    #print('\nMAE: {}'.format(scores_test[1]))
    #print('\nR^2: {}'.format(scores_test[2]))

    #final_preds = estimator.predict(X_test)
    #print(type(final_preds))

    final_preds = estimator.predict(X_test)
    print(type(final_preds))
    predicted_values = pd.DataFrame(final_preds)
    predicted_reversed = predicted_values.apply(reverse_transformer, args=(settings_array["PM10"],)) # check reversed and actual data if they are reversed or not
    Y_test = pd.DataFrame(Y_test)
    Y_reversed = Y_test.apply(reverse_transformer, args=(settings_array["PM10"],)) # check reversed and actual data if they are reversed or not

    mse = np.mean((predicted_reversed - Y_reversed) ** 2)

    rmse = sqrt(mean_squared_error(Y_reversed, predicted_reversed))

    print('Test MSE: ' + str(mse))
    print('Variance score: {:2f}'.format(r2_score(Y_reversed, predicted_reversed)))

    fig_verify = plt.figure()
    plt.plot(predicted_reversed, color="blue")
    plt.plot(Y_reversed, color="orange")
    plt.title('LSTM1 Train :' + begin_date + ' - ' + test_date  + '\nTest :' + begin_test + ' - ' + final_date)
    plt.ylabel('value')
    plt.xlabel('row')
    str_legend = 'station: '+ str(station) + '\nR2 = ' + str(r2_score(Y_reversed, predicted_reversed)) + '\nMSE = ' + str(mse.item())
    #str_data = 'Train :' + begin_date + ' - ' + test_date  + '\nTest :' + begin_test + ' - ' + final_date
    first_legend = plt.legend(['predicted', 'actual data'] , loc='upper right')
    #dates_legend = plt.legend([str_data], loc='upper center')


    ax = plt.gca().add_artist(first_legend)

    #ay = plt.gca().add_artist(dates_legend)


    plt.legend([AnyObject()], [str_legend],
               handler_map={AnyObject: AnyObjectHandler()})

    #plt.show()
    path_model_regression_verify = "./LSTM1_lille" + text + ".png"

    fig_verify.savefig(path_model_regression_verify)

    write_results(str_legend)
    os.remove(model_path_merged)



#test_station(salouel_raw_data, False) # MAE: 0.045106929216720444 R^2: 0.17245528477276584

dict = {'salouel': salouel_raw_data, 'roth': roth_raw_data, 'creil': creil_raw_data}


for key in dict.keys():
    test_station(dict[key], key, False)
    test_station(dict[key], key, True)


"""


for st in arr_row_data: # the best result is 2 -> saluel True
    test_station(st, False)
    test_station(st, True)
"""