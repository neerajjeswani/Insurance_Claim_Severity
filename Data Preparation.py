# Loading Data

import pandas as pd
import numpy as np

train = pd.read_csv("/Users/neerajjeswani/Desktop/All_State/train.csv")
test = pd.read_csv("/Users/neerajjeswani/Desktop/All_State/test.csv")

train.drop('Unnamed: 132', axis=1, inplace=True)
test.drop('Unnamed: 131', axis=1, inplace=True)


df = train.iloc[:,1:]

print(df.head(7))

# Data Statistics

print(df.shape)

print(df.describe())

print(df.skew())

# Transforming the loss data using log(1+x)

df['loss'] = np.log1p(df['loss'])

# Converting Categorical into Numerical Data

cols= df.columns

cat = 116
size = 15

labels=[]

for i in range(cat):
    t1 = df[cols[i]].unique()
    t2 = test[cols[i]].unique()
    labels.append(list(set(t1) or set(t2)))

del test
del t1
del t2

# One Hot Encoder

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

cats = []

for i in range(cat):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(df.iloc[:,i])
    feature = feature.reshape(df.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)

del feature

hot_cats =np.column_stack(cats)

print(hot_cats.shape)

df = np.concatenate((hot_cats,df.iloc[:,cat:].values), axis=1)

print(df.shape)

del cats
del hot_cats

# Data Preparation
    
X = df[:,:1241]
Y = df[:,1241]

del df

from sklearn import cross_validation
X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X, Y, test_size=.5, random_state=777)
del X
del Y

np.savetxt("/Users/neerajjeswani/Desktop/All_State/X_train.csv", X_train, delimiter=",",fmt='%f')
np.savetxt("/Users/neerajjeswani/Desktop/All_State/X_val.csv", X_val, delimiter=",",fmt='%f')
np.savetxt("/Users/neerajjeswani/Desktop/All_State/Y_train.csv", Y_train, delimiter=",",fmt='%f')
np.savetxt("/Users/neerajjeswani/Desktop/All_State/Y_Val.csv", Y_val, delimiter=",",fmt='%f')
