
import pandas as pd
import numpy as np
from matplotlib import pyplot
import seaborn as sns
from sklearn import preprocessing
import keras.backend as K
import keras
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM

from keras.layers.core import Activation
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Column_settings import *

#changed all , by . in cvs files

text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_results(text):
    file = open("results.txt", "w")

    now = datetime.datetime.now()

    with open("test.txt", "a") as file:
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


#print(salouel.info())



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
print("merged data goes here")
print(merged_data.info())

# Sorting by date, returns TrainModel, TestModel and Control dataframes
def separate_data(df, df_raw):

    df_raw = df_raw.iloc[1:]
    df_raw.columns = cols

    df["Date"] = pd.to_datetime(df_raw["Date"])
    train = df[df['Date'] < '31/12/2013'].drop("Date", axis=1)

    test = df[df['Date'] > '31/12/2013'].drop("Date", axis=1)
    return train, test


"""
REPLACE MISSING VALUES BY AVERAGE OF CORRESPONDING VALUES 


avg = np.nanmean([salouel_data.values, creil_data.values, roth_data.values], axis=0)

for df in [salouel_data, creil_data, roth_data]:
    df[df.isnull()] = avg

"""

"""
SEPARATE DF INTO TRAIN AND TEST DATA
TRAIN IS 2005 - 2013
TEST IS 2014 AND 2015

RETURN CORTÃ‰GE (TRAIN, TEST)
"""
#print("Number of columns: ")
#print(len(cols))

#print(salouel_raw_data.iloc[1:].head())

def separate_data(df, df_raw):

    df_raw = df_raw.iloc[1:]
    df_raw.columns = cols

    df["Date"] = pd.to_datetime(df_raw["Date"])
    train = df[df['Date'] < '31/12/2013'].drop("Date", axis=1)

    test = df[df['Date'] > '31/12/2013'].drop("Date", axis=1)
    return train, test




print("Train data: ")


label_array = np.array(salouel_data["PM10"])
#salouel_data = salouel_data.drop("PM10", axis=1)


#label_array = np.array(label_array)




"""
NORMALISATION
"""
#label_array = label_array.reshape(len(label_array), 1) # [ [y1] [y2] [y3]... ]


print("Label array goes here")

#print(label_array)
"""
min_max_scaler = preprocessing.MinMaxScaler()
#norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_salouel), columns=train_salouel.columns)
#norm_test_df = pd.DataFrame(min_max_scaler.fit_transform(test_salouel), columns=test_salouel.columns)
norm_df = pd.DataFrame(min_max_scaler.fit_transform(salouel_data), columns=salouel_data.columns)
norm_label_array = pd.DataFrame(min_max_scaler.fit_transform(label_array))
"""

print(salouel_data.head())
#for col in salouel_data.columns:
#    salouel_data[col] = salouel_data[col].apply(scale_transformer, args=(settings_array[col],))


print(salouel_data.head())

train_salouel_data, test_salouel_data = separate_data(salouel_data, salouel_raw_data)

#salouel_data["PM10"] = salouel_data["PM10"].apply(reverse_transformer, args=(settings_array["PM10"],))


print(train_salouel_data.head())




train_y_scaled_data = train_salouel_data['PM10']
train_x_scaled_data = train_salouel_data.drop("PM10", axis=1)

test_y_scaled_data = test_salouel_data['PM10']
test_x_scaled_data = test_salouel_data.drop("PM10", axis=1)


def make_plots(df):
    values = df.values
    groups = [ x for x in range(len(df.columns))]
    i = 1
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(df.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()


make_plots(test_x_scaled_data)


