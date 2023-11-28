import pandas as pd
import numpy as np

df = pd.read_csv("MissingData1.txt", sep='\t', header = None)
df.replace(1.00000000000000e+99, np.nan, inplace=True)

df[0].fillna(df[0].mean(), inplace=True)
df[1].fillna(df[1].median(), inplace=True)
df[2].fillna(df[2].mean(), inplace=True)
df[3].fillna(df[3].median(), inplace=True)
df[5].fillna(df[5].median(), inplace=True)
df[6].fillna(df[6].median(), inplace=True)
df[7].fillna(df[7].median(), inplace=True)
df[8].fillna(df[8].mean(), inplace=True)
df[9].fillna(df[9].mean(), inplace=True)
df[10].fillna(df[10].mean(), inplace=True)
df[11].fillna(df[11].mean(), inplace=True)
df[12].fillna(df[12].median(), inplace=True)
df[13].fillna(df[13].mean(), inplace=True)

df.to_csv('NewData1.txt', sep='\t', index = False)