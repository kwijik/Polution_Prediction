from pandas import DataFrame
from pandas import read_csv

df = read_csv("Moyennes_J_roth_2005_2015.csv", header=None, sep=';', decimal=',')
df = df.iloc[1:,]
#print(df.head())
date = df.iloc[:,0]

df = df.drop([0], axis=1)
df = df.astype(float)
#print(df.head())
df[0] = date
df.dropna()
print(df.head())

df.sort_index(axis=1, inplace=True)

print()
print(df.head())

# transform in floats
df.to_csv("df_roth.csv", index=False)