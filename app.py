

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

from keras.layers.core import Activation

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



#changed all , by . in cvs files


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
REPLACE MISSING VALUES BY AVERAGE OF CORRESPONDING VALUES 
"""

avg = np.nanmean([salouel_data.values, creil_data.values, roth_data.values], axis=0)

for df in [salouel_data, creil_data, roth_data]:
    df[df.isnull()] = avg

print("Salouel:")
print(salouel_data.head())
print("/*/*/*/*")

print("Creil:")
print(creil_data.head())
print("/*/*/*/*")

print("Roth:")
print(roth_data.head())
print("/*/*/*/*")


"""
SEPARATE DF INTO TRAIN AND TEST DATA
TRAIN IS 2005 - 2013
TEST IS 2014 AND 2015

RETURN CORTÉGE (TRAIN, TEST)
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
train_salouel, test_salouel = separate_data(salouel_data, salouel_raw_data)
label_array = np.array(train_salouel["PM10"])
train_salouel.drop("PM10", axis=1)

label_array = np.array(label_array)
print(test_salouel.head())


"""
NORMALISATION
"""

min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_salouel), columns=train_salouel.columns)
norm_test_df = pd.DataFrame(min_max_scaler.fit_transform(test_salouel), columns=test_salouel.columns)

sequence_length = 50


def gen_sequence(df):
    X_train = []
    for i in range(sequence_length, len(df)):
        X_train.append(df.iloc[i - sequence_length : i, :].values)
    return X_train

def  gen_y(df):
    Y_train = []
    for i in range(sequence_length, len(df)):
        Y_train.append(df.loc[i - sequence_length : i])
    return Y_train

seq_array = gen_sequence(norm_train_df)
Y_train = np.array(gen_y(train_salouel["PM10"]))
"""


X_train = []
for i in range(sequence_length, len(norm_train_df) - sequence_length):
    X_train.append(norm_train_df[i - sequence_length:i, :])

"""

print("length: ")
print(norm_train_df.head())
seq_array = np.array(seq_array)
print("seq: ")
print(seq_array[0])


def r2_keras(y_true, y_pred):
    """Coefficient of Determination
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

nb_features = seq_array.shape[2]

print("nb_features: ")
print(nb_features)

label_array = label_array.reshape(len(label_array), 1)

nb_out = label_array.shape[1]


print("label array : ")
print(label_array)

model = Sequential()
model.add(LSTM(
         input_shape=(sequence_length, nb_features),
         units=100,
         return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(
          units=50,
          return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.add(Activation("linear"))
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae', r2_keras])

print(model.summary())


print("Length seq_array ")
print(len(seq_array))

print("Length of each seq_array ")
print(len(seq_array[0]))

print("Length of each seq_array ")
print(len(seq_array[0]))


print("Model of each sub seq_array ")
print(seq_array[0])


print("The shape is : ")
print(seq_array.shape)

model_path = './regression_model.h5'
print("Type of the X")
print(type(seq_array))

print("Length of each Y_train ")
print(len(label_array[0]))

print("type of the Y: ")
print(type(label_array))


print("The Y_shape is : ")
print(label_array.shape)
label_array = label_array[0:len(label_array) - sequence_length]

history = model.fit(seq_array, label_array, epochs=15, batch_size=200, validation_split=0.05, verbose=2,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
          )






