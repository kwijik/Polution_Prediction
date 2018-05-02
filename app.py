

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
Y is PM10
NO2 et O3 not really useful
"""

plt.hist(salouel_data['RR'], bins=30)
plt.ylabel("quantity")
plt.xlabel('value')
plt.title('precipitations ')

plt.savefig("RR.png")

sns.distplot(salouel_data['RR'])


#sns_plot = sns.distplot(salouel_data['RR'])
#sns_plot.savefig("RR.png")
salouel_data['RR']
#sns.set(color_codes=True)
#sns.distplot(salouel_data['RR'])
#sns.figure.savefig("RR.png")
"""
print("ddddd")
print(salouel_data['DXY'].get_name)


def generate_plots(df):
    columns = df.columns
    
    #for col in columns:
    #    print(col)

    
    for col in columns:
        plt.hist(df[col].values, bins=30)
        plt.ylabel("quantity")
        plt.xlabel('value')
        name = col + 'png'
        plt.savefig(name)

generate_plots(salouel_data)

"""

print(" salouel stats ")

print(salouel_data['RR'].describe())

def stats(df):
    mean = df.mean()
print(salouel_data.columns[1])



print(salouel_data.info())


np.random.seed(1234)
PYTHONHASHSEED = 0

model_path = 'regression_model.h5'


def prep_dt(df):
    df = df.iloc[1:]
    df.columns = cols
    dates = df["Date"]

    df = df.drop(["Date", 'NO2', 'O3'], axis=1)
    df = df.astype(float)
    df["Date"] = dates
    print(dates)
    return df

#train = salouel_raw_data[salouel_raw_data['Date'] < '31/12/2013']
print("DATA: ")


salouel_raw_data = salouel_raw_data.iloc[1:]
salouel_raw_data.columns = cols

dates = salouel_raw_data["Date"]
salouel_data["Dates"] = dates



salouel_data['Dates'] = pd.to_datetime(salouel_data['Dates'])
print(salouel_data.info())


train = salouel_data[salouel_data['Dates'] < '31/12/2013']

print(train.info())

test = salouel_data[salouel_data['Dates'] > '31/12/2013']

print(test.head())

