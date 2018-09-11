import pandas as pd
import numpy as np
import datetime


from pollution_plots import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(7)


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

print(salouel_raw_data.info())