
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

from Column_settings import *

#changed all , by . in cvs files

text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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

def merge_data(array_raw_data):
    arr_dfs = []
    # concat is first
    for df in array_raw_data:
        df = df.iloc[1:]
        df.columns = cols
        df_temp = df.drop(["Date", 'NO2', 'O3'], axis=1)
        df_temp = df_temp.astype(float)
        df_temp["Date"] = pd.to_datetime(df["Date"])
        arr_dfs.append(df_temp)
    cnc_df = pd.concat(arr_dfs).sort_values("Date").dropna(axis=0, how="any")
    cnc_df = cnc_df[cnc_df['Date'] > '31/12/2010']
    train = cnc_df[cnc_df['Date'] < '31/12/2013'].drop("Date", axis=1)
    test = cnc_df[cnc_df['Date'] > '31/12/2013'].drop("Date", axis=1)
    return train, test



def separate_data(df, df_raw):

    df_raw = df_raw.iloc[1:]
    df_raw.columns = cols

    df["Date"] = pd.to_datetime(df_raw["Date"])
    train = df[df['Date'] < '31/12/2014'].drop("Date", axis=1)

    test = df[df['Date'] > '31/12/2014'].drop("Date", axis=1)
    return train, test


train_merge, test_merge = merge_data([salouel_raw_data, creil_raw_data, roth_raw_data])



"""
NORMALISATION
"""

for col in salouel_data.columns:
    salouel_data[col] = salouel_data[col].apply(scale_transformer, args=(settings_array[col],))


for col in train_merge.columns:
    train_merge[col] = train_merge[col].apply(scale_transformer, args=(settings_array[col],))


for col in test_merge.columns:
    test_merge[col] = test_merge[col].apply(scale_transformer, args=(settings_array[col],))




train_salouel_data, test_salouel_data = separate_data(salouel_data, salouel_raw_data)

#salouel_data["PM10"] = salouel_data["PM10"].apply(reverse_transformer, args=(settings_array["PM10"],))





train_y_merged_scaled_data = train_merge['PM10']
train_x_merged_scaled_data = train_merge.drop("PM10", axis=1)
test_y_merged_scaled_data = test_merge['PM10']
test_x_merged_scaled_data = test_merge.drop("PM10", axis=1)

train_y_scaled_data = train_salouel_data['PM10']
train_x_scaled_data = train_salouel_data.drop("PM10", axis=1)
test_y_scaled_data = test_salouel_data['PM10']
test_x_scaled_data = test_salouel_data.drop("PM10", axis=1)

n_past = 90
n_future = 0

def gen_sequence(df):
    X_seq = []
    for i in range(n_past, len(df) - n_future + 1):
        X_seq.append(df.iloc[i - n_past : i, :].values)
    return X_seq

# [ [15], [10], [20] ]

#seq_array = gen_sequence(norm_train_df)
#Y_train = np.array(gen_y(train_salouel["PM10"]))

col_string = " ".join(str(x) for x in train_x_scaled_data.columns)

print("features sunt: ")
print(col_string)


train_seq_merged_df = np.array(gen_sequence(train_x_merged_scaled_data))
test_seq_merged_df = np.array(gen_sequence(test_x_merged_scaled_data))


train_seq_df = gen_sequence(train_x_scaled_data)
test_seq_df = gen_sequence(test_x_scaled_data)
train_seq_np = np.array(train_seq_df)
test_seq_np = np.array(test_seq_df)

#y_scaled_data

train_y_merged_scaled_data = train_y_merged_scaled_data.values.reshape(len(train_y_merged_scaled_data), 1) # [ [] [] []]
test_y_merged_scaled_data = test_y_merged_scaled_data.values.reshape(len(test_y_merged_scaled_data), 1)


train_y_scaled_data = train_y_scaled_data.values.reshape(len(train_y_scaled_data), 1) # [ [] [] []]
test_y_scaled_data = test_y_scaled_data.values.reshape(len(test_y_scaled_data), 1)

nb_features = train_seq_np.shape[2]

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
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae', r2_keras])

print(model.summary())


#model_path = './regression_model.h5'

model_path_merged = './regression_model_merged.h5'


train_merged_label_array = train_y_merged_scaled_data[n_past + n_future - 1:len(train_y_merged_scaled_data)] # Ð¿Ñ€Ð¾Ð²ÐµÑ€Ð¸Ñ‚ÑŒ Ñ€Ð°Ð·Ð¼ÐµÑ€ (+-1)!
test_merged_y_scaled_data = test_y_merged_scaled_data[n_past + n_future - 1:len(test_y_merged_scaled_data)] # Ð¿Ñ€Ð¾Ð²ÐµÑ€Ð¸Ñ‚ÑŒ Ñ€Ð°Ð·Ð¼ÐµÑ€ (+-1)!

train_label_array = train_y_merged_scaled_data[n_past + n_future - 1:len(train_y_merged_scaled_data)] # Ð¿Ñ€Ð¾Ð²ÐµÑ€Ð¸Ñ‚ÑŒ Ñ€Ð°Ð·Ð¼ÐµÑ€ (+-1)!
test_y_scaled_data = test_y_merged_scaled_data[n_past + n_future - 1:len(test_y_merged_scaled_data)] # Ð¿Ñ€Ð¾Ð²ÐµÑ€Ð¸Ñ‚ÑŒ Ñ€Ð°Ð·Ð¼ÐµÑ€ (+-1)!


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
scores_test = estimator.evaluate(np.array(train_seq_merged_df), np.array(train_merged_label_array), verbose=2)
print('\nMAE: {}'.format(scores_test[1]))
print('\nR^2: {}'.format(scores_test[2]))

"""

history = model.fit(train_seq_np, train_label_array, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_seq_np, test_y_scaled_data), verbose=2,
          callbacks = [#keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
          )




history = model.fit(seq_array, label_array, epochs=30, batch_size=200, validation_split=0.05, verbose=2,
          callbacks = [#keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
          )

"""

#print(predicted_reversed)

text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")




"""


fig_acc = plt.figure(figsize=(10, 10))
text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
plt.plot(history.history['r2_keras'])
plt.plot(history.history['val_r2_keras'])
plt.title('model r^2')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
path_r2 = "./model_r2_" + text + ".png"
fig_acc.savefig(path_r2)


"""


"""

# summarize history for MAE
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
path_mae = "./model_mae_" + text + ".png"
fig_acc.savefig(path_mae)

# summarize history for Loss
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
path_reg_loss = "./model_regression_loss_" + text + ".png"
fig_acc.savefig(path_reg_loss)
"""


"""

####
# Creil data
###
print("Creil data:")

# Scaling goes here

print(creil_data.head())
for col in creil_data.columns:
    creil_data[col] = creil_data[col].apply(scale_transformer, args=(settings_array[col],))

# Test and Train
train_creil_data, test_creil_data = separate_data(creil_data, creil_raw_data)

# Creation of X and Y
cr_y_scaled_data = test_creil_data['PM10']
cr_x_sc = test_creil_data.drop("PM10", axis=1)
print("X data ")
print(cr_x_sc.head())

print("Y data")
print(cr_y_scaled_data.head())


# Generation of sequences
cr_seq = gen_sequence(cr_x_sc)

# Transformation X in numpy array
cr_seq = np.array(cr_seq)


# Reshape of Y
cr_y_scaled_data = cr_y_scaled_data.reshape(len(cr_y_scaled_data), 1)

# Optimizing length of Y
cr_y_scaled_data = cr_y_scaled_data[n_past + n_future - 1:len(cr_y_scaled_data)]

"""


####
# Roth data
####
from math import floor
print("Creil data:")

# Scaling goes here

print(roth_data.head())
for col in roth_data.columns:
    roth_data[col] = roth_data[col].apply(scale_transformer, args=(settings_array[col],))

# Test and Train
#train_creil_data, test_creil_data = separate_data(creil_data, creil_raw_data)

separate_percent = 0.05 # 20% will be separated

test_number = floor(len(roth_data) * separate_percent) # 3772 * 005 = 188
print("total number of lines " + str(len(roth_data))) # 3772
print("number of tested lines " + str(test_number)) # 188
data_roth = roth_data.loc[len(roth_data) - test_number : len(roth_data) - 1, :] # 397
print("Roth ")
print(len(data_roth))



# Creation of X and Y
cr_y_scaled_data = data_roth['PM10']
cr_x_sc = data_roth.drop("PM10", axis=1)
print("X data ")
print(cr_x_sc.head())

print("Y data")
print(cr_y_scaled_data.head())


# Generation of sequences
cr_seq = gen_sequence(cr_x_sc)

# Transformation X in numpy array
cr_seq = np.array(cr_seq)


# Reshape of Y
cr_y_scaled_data = cr_y_scaled_data.values.reshape(len(cr_y_scaled_data), 1)

# Optimizing length of Y
cr_y_scaled_data = cr_y_scaled_data[n_past + n_future - 1:len(cr_y_scaled_data)]

"""


print("Shape of train_seq_np")
print(np.array(train_seq_np).shape)

print("Shape of train_label_array")
print(np.array(train_label_array).shape)

print("Shape of cr_seq")
print(cr_seq.shape)

print("Shape of test_seq_np")
print(test_seq_np.shape)

print("Shape of test_y_scaled_data")
print(test_y_scaled_data.shape)

if os.path.isfile(model_path):
#if os.path.isfile(model_path_merged):
    #
    ##
    ###
    ######
    estimator = load_model(model_path, custom_objects={'r2_keras': r2_keras})

    scores_test = estimator.evaluate(np.array(train_seq_np), np.array(train_label_array), verbose=2)
    print('\nMAE: {}'.format(scores_test[1]))
    print('\nR^2: {}'.format(scores_test[2]))

    y_pred_test = estimator.predict(cr_seq)

    y_true_test = cr_y_scaled_data


    predicted_values = pd.DataFrame(y_pred_test)

    predicted_reversed = predicted_values.apply(reverse_transformer, args=(settings_array["PM10"],))

    predicted_reversed = pd.DataFrame(predicted_reversed)


    # RMSE is calculated on the test values (predicted and real)

    y_true_test = pd.DataFrame(y_true_test)

    true_test_reversed = y_true_test.apply(reverse_transformer, args=(settings_array["PM10"],))

    rmse = sqrt(mean_squared_error( true_test_reversed, predicted_reversed ))
    print('Test RMSE: %.3f' % rmse)

    res = "Sequence length: " + str(n_past) + "\nNombre d'epochs: " + str(EPOCHS) + "\nBatch_size: " + str(BATCH_SIZE) + '\n' + col_string + '\nMAE: {}'.format(scores_test[1]) + '\nR^2: {}'.format(scores_test[2]) + '\nTest RMSE: %.3f' % rmse
    write_results(res)

    #print("Y true values: ")
   # print(y_true_test[0])
    name_predicted = './' + text + '_predicted_submit_test.csv'
    name_real = './' + text + '_true_submit_test.csv'
    true_test_reversed.to_csv(name_real, index=None)
    predicted_reversed.to_csv(name_predicted, index=None)


    # Plot in blue color the predicted data and in green color the
    # actual data to verify visually the accuracy of the model.
    fig_verify = plt.figure(figsize=(100, 50))
    plt.plot(predicted_reversed, color="blue")
    plt.plot(true_test_reversed, color="orange")
    plt.title('prediction')
    plt.ylabel('value')
    plt.xlabel('row')
    plt.legend(['predicted', 'actual data'], loc='upper left')
    plt.show()
    path_model_regression_verify = "./model_regression_verify_" + text + ".png"

    fig_verify.savefig(path_model_regression_verify)
    os.remove("./regression_model.h5")
"""