import pandas as pd
import numpy as np
import datetime
import time
from pandas import concat
from pandas import DataFrame
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(7)
from sklearn.metrics import r2_score
from pollution_plots import *
import matplotlib.patches as mpatches

folder = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
np.random.seed(7)



# Function to plot the data
# input: final final_preds_expand - np array, test_y_expand - array, str_legend - string, station - string
# output: void
def plot_test(final_preds_expand, test_y_expand, str_legend, station):
    text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig_verify = plt.figure(figsize=(17,8))
    plt.plot(final_preds_expand, color="blue")
    plt.plot(test_y_expand, color="orange")
    plt.title('AV Seq2seq : ' + str(station))
    plt.ylabel('Pollution [PM10]')
    plt.xlabel('Daily timestep for 1 year')
    first_legend = plt.legend(['predicted', 'actual data'] , loc='upper left')
    dates_legend = plt.legend([str_legend], loc='upper right')
    ax = plt.gca().add_artist(first_legend)
    plt.show()
    path_model_regression_verify = "./Images/s2s_Creil.png"
    fig_verify.savefig(path_model_regression_verify)

#

# Saves results
# input text - string
# output - void
def write_results(text):
    #file = open("results.txt", "w")

    now = datetime.datetime.now()

    with open("grid_seq.txt", "a") as file:
        text = now.strftime("%Y-%m-%d %H:%M:%S") + ": " + "\n" + text + "\n" + "\n"
        file.write(text)

# reading from file
salouel_raw_data = pd.read_csv("./Data/salou.csv", header=None, sep=';', decimal=',')
roth_raw_data = pd.read_csv("./Data/roth.csv", header=None, sep=';', decimal=',')
creil_raw_data = pd.read_csv("./Data/creil.csv", header=None, sep=';', decimal=',')


cols = ['Date', 'PM10', 'RR', 'TN', 'TX', 'TM',
                     'PMERM', 'FFM', 'UM', 'GLOT', 'DXY']


arr_row_data = [salouel_raw_data, creil_raw_data, roth_raw_data]

#t2 = pd.read_csv('t2.csv')
"""
Deleting dates column and making it all floats
"""

# converts to float
# input: df - dataframe
# output dataframe
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

#for col in salouel_data.columns:
    #salouel_data[col] = salouel_data[col].apply(scale_transformer, args=(settings_array[col],))


# Transforms month on string
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

# adds lagged columns
# input: data - dataframe, n_in - int, n_out - int, dropnan - boolean
# output: dataframe
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    print('in series to suprsd')
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    print(df.info())
    print(df.head(1))
    if dropnan:
        df.dropna(inplace=True)
    print(df.info())
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
    agg.columns = names
    # drop rows with NaN values

    if dropnan:
        agg.dropna(inplace = True)

    return agg

# Divide data into Test, Train and Validation
# input: df_temp - dataframe, keep_season - boolean, keep_wind - boolean
# output: 3 dataframes
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
    #train_data.to_csv('./data/train.csv')

    #test_data.to_csv('./data/test.csv')

    print("Dataframes saved...")
    return train_data, test_data, validation_data


# Transforms categorical variables in numerique variables
# input: dataframe
# output: dataframe
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

# main function
# input: data - dataframe, station - string
# output: void
def test_station(data, station):
    global folder, os
    print("Entered in Function " + station)
    #text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not (os.path.isdir(folder)):
        os.mkdir(folder)
    df_temp = clean_data(data)
    # display_plot(df_temp)

    # creating train, test and val-n data
    train_data, test_data, validation_data = prepare_data(df_temp, True, True)
    # AV
    X_train = train_data.loc[:, ['PM10', 'RR', 'TM', 'PMERM', 'FFM', 'UM', 'GLOT', 'TN', 'TX']].values.copy()
    X_test = test_data.loc[:, ['PM10', 'RR', 'TM', 'PMERM', 'FFM', 'UM', 'GLOT', 'TN', 'TX']].values.copy()


    # datapreprocesing
    reframed_train = series_to_supervised(X_train, n_out=1)
    reframed_test = series_to_supervised(X_test, n_out=1)
    X_test = np.copy(reframed_test.values)
    X_train = np.copy(reframed_train)
    y_train = train_data['PM10'].values.copy().reshape(-1, 1)
    y_test = test_data['PM10'].values.copy().reshape(-1, 1)

    #Scaling data
    for i in range(X_train.shape[1]):
        temp_mean = X_train[:, i].mean()
        temp_std = X_train[:, i].std()
        X_train[:, i] = (X_train[:, i] - temp_mean) / temp_std
        X_test[:, i] = (X_test[:, i] - temp_mean) / temp_std


    ## z-score transform y
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    input_seq_len = 1
    output_seq_len = 1
    batch_size = 16

    # generate train seqences
    # input: x - np array, y - np array, batch_size - int, input_seq_len - int, output_seq_len - int
    # output: 2 np arrays
    def generate_train_samples(x=X_train, y=y_train, batch_size=batch_size, input_seq_len=input_seq_len,
                               output_seq_len=output_seq_len):
        total_start_points = len(x) - input_seq_len - output_seq_len
        start_x_idx = np.random.choice(range(total_start_points), batch_size, replace=False)

        input_batch_idxs = [list(range(i, i + input_seq_len)) for i in start_x_idx]
        input_seq = np.take(x, input_batch_idxs, axis=0)

        output_batch_idxs = [list(range(i + input_seq_len, i + input_seq_len + output_seq_len)) for i in start_x_idx]
        output_seq = np.take(y, output_batch_idxs, axis=0)

        return input_seq, output_seq  # in shape: (batch_size, time_steps, feature_dim)

    # generate test seqences
    # input: x - np array, y - np array, batch_size - int, input_seq_len - int
    # output: 2 np arrays
    def generate_test_samples(x=X_test, y=y_test, input_seq_len=input_seq_len, output_seq_len=output_seq_len):
        total_samples = x.shape[0]

        input_batch_idxs = [list(range(i, i + input_seq_len)) for i in
                            range((total_samples - input_seq_len - output_seq_len))]
        input_seq = np.take(x, input_batch_idxs, axis=0)

        output_batch_idxs = [list(range(i + input_seq_len, i + input_seq_len + output_seq_len)) for i in
                             range((total_samples - input_seq_len - output_seq_len))]
        output_seq = np.take(y, output_batch_idxs, axis=0)

        return input_seq, output_seq

    x, y = generate_train_samples()

    test_x, test_y = generate_test_samples()

    from tensorflow.contrib import rnn
    from tensorflow.python.ops import variable_scope
    from tensorflow.python.framework import dtypes
    import tensorflow as tf
    import copy
    import os

    ## Parameters
    learning_rate = 0.02
    lambda_l2_reg = 0.003

    ## Network Parameters
    # length of input signals
    input_seq_len = input_seq_len
    # length of output signals
    output_seq_len = output_seq_len
    # size of LSTM Cell
    hidden_dim = 20
    # num of input signals
    input_dim = X_train.shape[1]
    # num of output signals
    output_dim = y_train.shape[1]
    # num of stacked lstm layers
    num_stacked_layers = 2
    # gradient clipping - to avoid gradient exploding
    GRADIENT_CLIPPING = 2.5

    # seq2seq architecture
    # input: boolean
    # output: creates seq3seq model
    def build_graph(feed_previous=False):
        print("in build_graph " + station)

        tf.reset_default_graph()

        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        weights = {
            'out': tf.get_variable('Weights_out', \
                                   shape=[hidden_dim, output_dim], \
                                   dtype=tf.float32, \
                                   initializer=tf.truncated_normal_initializer()),
        }
        biases = {
            'out': tf.get_variable('Biases_out', \
                                   shape=[output_dim], \
                                   dtype=tf.float32, \
                                   initializer=tf.constant_initializer(0.)),
        }

        with tf.variable_scope('Seq2seq'):
            # Encoder: inputs
            enc_inp = [
                tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
                for t in range(input_seq_len)
            ]

            # Decoder: target outputs
            target_seq = [
                tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
                for t in range(output_seq_len)
            ]

            # Give a "GO" token to the decoder.
            # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the
            # first element will be fed as decoder input which is then 'un-guided'
            dec_inp = [tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO")] + target_seq[:-1]

            with tf.variable_scope('LSTMCell'):
                cells = []
                for i in range(num_stacked_layers):
                    with tf.variable_scope('RNN_{}'.format(i)):
                        cells.append(tf.contrib.rnn.LSTMCell(hidden_dim))
                cell = tf.contrib.rnn.MultiRNNCell(cells)

            def _rnn_decoder(decoder_inputs,
                             initial_state,
                             cell,
                             loop_function=None,
                             scope=None):

                with variable_scope.variable_scope(scope or "rnn_decoder"):
                    state = initial_state
                    outputs = []
                    prev = None
                    for i, inp in enumerate(decoder_inputs):
                        if loop_function is not None and prev is not None:
                            with variable_scope.variable_scope("loop_function", reuse=True):
                                inp = loop_function(prev, i)
                        if i > 0:
                            variable_scope.get_variable_scope().reuse_variables()
                        output, state = cell(inp, state)
                        outputs.append(output)
                        if loop_function is not None:
                            prev = output
                return outputs, state

            def _basic_rnn_seq2seq(encoder_inputs,
                                   decoder_inputs,
                                   cell,
                                   feed_previous,
                                   dtype=dtypes.float32,
                                   scope=None):

                with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
                    enc_cell = copy.deepcopy(cell)
                    _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
                    if feed_previous:
                        return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
                    else:
                        return _rnn_decoder(decoder_inputs, enc_state, cell)

            def _loop_function(prev, _):
                '''Naive implementation of loop function for _rnn_decoder. Transform prev from
                dimension [batch_size x hidden_dim] to [batch_size x output_dim], which will be
                used as decoder input of next time step '''
                return tf.matmul(prev, weights['out']) + biases['out']

            dec_outputs, dec_memory = _basic_rnn_seq2seq(
                enc_inp,
                dec_inp,
                cell,
                feed_previous=feed_previous
            )

            reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]

        # Training loss and optimizer
        with tf.variable_scope('Loss'):
            # L2 loss
            output_loss = 0
            for _y, _Y in zip(reshaped_outputs, target_seq):
                output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

            # L2 regularization for weights and biases
            reg_loss = 0
            for tf_var in tf.trainable_variables():
                if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

            loss = output_loss + lambda_l2_reg * reg_loss

        with tf.variable_scope('Optimizer'):
            optimizer = tf.contrib.layers.optimize_loss(
                loss=loss,
                learning_rate=learning_rate,
                global_step=global_step,
                optimizer='Adam',
                clip_gradients=GRADIENT_CLIPPING)

        saver = tf.train.Saver

        return dict(
            enc_inp=enc_inp,
            target_seq=target_seq,
            train_op=optimizer,
            loss=loss,
            saver=saver,
            reshaped_outputs=reshaped_outputs,
        )

    total_iteractions = 180
    batch_size = 16

    x = np.linspace(0, 40, 130)
    train_data_x = x[:110]

    rnn_model = build_graph(feed_previous=False)

    saver = tf.train.Saver()
    start_train = time.time()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        print("Training losses: ")
        for i in range(total_iteractions):
            batch_input, batch_output = generate_train_samples(batch_size=batch_size)

            feed_dict = {rnn_model['enc_inp'][t]: batch_input[:, t] for t in range(input_seq_len)}
            feed_dict.update({rnn_model['target_seq'][t]: batch_output[:, t] for t in range(output_seq_len)})
            _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)


        temp_saver = rnn_model['saver']()
        sous_chemin = './' + folder + "/"

        save_path = temp_saver.save(sess, os.path.join(sous_chemin, 'mv_ts_pollution_case'))

    print("Checkpoint saved at: ", save_path)

    end_train = time.time()

    print('Time taken to train is {} minutes'.format((end_train - start_train) / 60))
    print('Time in seconds is: {}'.format(end_train - start_train))

    rnn_model = build_graph(feed_previous=True)

    start_test = time.time()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sous_chemin = './' + folder + "/"
        saver = rnn_model['saver']().restore(sess, os.path.join(sous_chemin, 'mv_ts_pollution_case'))

        feed_dict = {rnn_model['enc_inp'][t]: test_x[:, t, :] for t in range(input_seq_len)}  # batch prediction
        feed_dict.update({rnn_model['target_seq'][t]: np.zeros([test_x.shape[0], output_dim], dtype=np.float32) for t in
                          range(output_seq_len)})
        final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)
        final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
        final_preds = np.concatenate(final_preds, axis=1)


    # Unscale the predictions
    # make into one array
    end_test = time.time()

    #print('Time taken is: {} minutes.'.format((end_test - start_test) / 60))
    #print('Time in seconds is: {}'.format(end_test - start_test))

    test_time = end_test - start_test

    dim1, dim2 = final_preds.shape[0], final_preds.shape[1]

    preds_flattened = final_preds.reshape(dim1 * dim2, 1)

    unscaled_yhat = pd.DataFrame(preds_flattened, columns=['PM10']).apply(lambda x: (x * y_std) + y_mean)

    yhat_inv = unscaled_yhat.values

    test_y_flattened = test_y.reshape(dim1 * dim2, 1)

    unscaled_y = pd.DataFrame(test_y_flattened, columns=['PM10']).apply(lambda x: (x * y_std) + y_mean)

    y_inv = unscaled_y.values

    pd.concat((unscaled_y, unscaled_yhat), axis=1)

    mse = np.mean((yhat_inv - y_inv) ** 2)

    rmse = np.sqrt(mse)
    str_legend = 'R2 = ' + str(r2_score(y_inv, yhat_inv)) + '\nMSE = ' + str(mse.item()) + '\nRMSE = ' + str(rmse)  + '\nInput=1'  + '; Output='+ str(otp)

    plot_test(y_inv, yhat_inv, str_legend, station)

    print('Root Mean Squared Error: {:.4f}'.format(rmse))

    # Calculate R^2 (regression score function)
    test_R2 = r2_score(y_inv, yhat_inv)
    print('Variance score: {:2f}'.format(r2_score(y_inv, yhat_inv)))
    if os.path.isfile("./regression_model_merged.h5"):
        os.remove("./regression_model_merged.h5")


test_station(creil_raw_data, 'Creil')
