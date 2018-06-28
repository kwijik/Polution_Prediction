

import pandas as pd
import numpy as np

salouel_raw_data = pd.read_csv("Moyennes_J_salouel_2005_2015.csv", header=None, sep=';', decimal=',')


creil_raw_data = pd.read_csv("Moyennes_J_creil_2005_2015.csv", header=None, sep=';', decimal=',')
roth_raw_data = pd.read_csv("Moyennes_J_roth_2005_2015.csv", header=None, sep=';', decimal=',')

arr = [salouel_raw_data, creil_raw_data, roth_raw_data]
test_date = '31/12/2014'


cols = ['Date', 'NO2', 'O3', 'PM10', 'RR', 'TN', 'TX', 'TM',
                     'PMERM', 'FFM', 'DXY', 'UM', 'GLOT']

def gen_test_data(array_raw_data):
    # concat is first
    df = array_raw_data.iloc[1:]
    df.columns = cols
    df_temp = df.drop(["Date", 'NO2', 'O3'], axis=1)
    df_temp = df_temp.astype(float)
    df_temp["Date"] = pd.to_datetime(df["Date"])

    df_temp = df_temp[df_temp['Date'] > test_date].dropna(axis=0, how="any")

    Y = df_temp['PM10']
    X = df_temp[['Date', 'RR', 'TN', 'TX', 'TM',
                     'PMERM', 'FFM', 'DXY', 'UM', 'GLOT']]
    return Y, X

Y, X = gen_test_data(creil_raw_data)

print(X.info())


res = (next(i for i, j in enumerate(arr) if j is roth_raw_data) + 1) % len(arr)  # 1

print(res)


"""


next_el = arr[(arr.index(roth_raw_data) +1) % 3 ]

print(next_el.head(2))

print("-------")

print(salouel_raw_data.head(2))
"""