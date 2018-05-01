

import pandas as pd
import numpy as np

#changed all , by . in cvs files


salouel_raw_data = pd.read_csv("Moyennes_J_salouel_2005_2015.csv", header=None, sep=';', decimal=',')


creil_raw_data = pd.read_csv("Moyennes_J_creil_2005_2015.csv", header=None, sep=';', decimal=',')

cols = ['Date', 'NO2', 'O3', 'PM10', 'RR', 'TN', 'TX', 'TM',
                     'PMERM', 'FFM', 'DXY', 'UM', 'GLOT']

roth_raw_data = pd.read_csv("Moyennes_J_roth_2005_2015.csv", header=None, sep=';', decimal=',')


#print(salouel.info())


def prepare_data(df):
    df = df.iloc[1:]
    df.columns = cols
    df = df.drop("Date", axis=1)
    df = df.astype(float)
    return df

"""
TODO:
def prepare_data_ar(array_of_df)
"""

salouel_data = prepare_data(salouel_raw_data)
creil_data = prepare_data(creil_raw_data)
roth_data = prepare_data(roth_raw_data)


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
