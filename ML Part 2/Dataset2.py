import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


df1 = pd.read_csv("MissingData2.txt", sep='\t', header = None)
df1.replace(1.00000000000000e+99, np.nan, inplace=True)

#Using KNN to fill the missing values
knn_imputer = KNNImputer(n_neighbors=3)

df1 = pd.DataFrame(knn_imputer.fit_transform(df1), columns=df1.columns)

df1.to_csv('NewData2.txt', sep='\t', index = False)
