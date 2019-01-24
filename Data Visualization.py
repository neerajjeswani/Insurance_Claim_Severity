# Loading Data

import pandas as pd
import numpy as np

train = pd.read_csv("/Users/neerajjeswani/Desktop/All_State/train.csv")
test = pd.read_csv("/Users/neerajjeswani/Desktop/All_State/test.csv")

train.drop('Unnamed: 132', axis=1, inplace=True)
test.drop('Unnamed: 131', axis=1, inplace=True)


df = train.iloc[:,1:]

# Data Visualization

import matplotlib.pyplot as plt
import seaborn as sns

cat = 116
size = 15

data = df.iloc[:,cat:]

cols = data.columns

#Plotting Continuous Variables

fig, axes = plt.subplots(nrows=14, ncols=1, figsize=(7, 120))
for i in range(14):
    sns.violinplot(y=cols[i], data=df, ax=axes[i])

#Plotting Loss

sns.violinplot(y='loss', data=df)

# Transforming the loss data using log(1+x)

df['loss'] = np.log1p(df['loss'])

sns.violinplot(y='loss', data=df)

corr = data.corr()
del data
plt.figure(figsize=(16,8))
plt.matshow(corr, fignum=1, cmap='Reds')

corr_list = []

for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (corr.iloc[i,j] >= 0.5 and corr.iloc[i,j] < 1) or (corr.iloc[i,j] < 0 and corr.iloc[i,j] <= -0.5):
            corr_list.append([corr.iloc[i,j],i,j])

sorted_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

for x,y,z in sorted_corr_list:
    print ("%s and %s = %.2f" % (cols[y],cols[z],x))
    

for x,y,z in sorted_corr_list:
    sns.pairplot(df, size=7, x_vars=cols[y],y_vars=cols[z] )
    plt.show()

# Categorical Data Visualization
    
cols= df.columns

for i in range(29):
    fig,axes = plt.subplots(nrows=1,ncols=4,figsize=(12, 8))
    for j in range(4):
        sns.countplot(x=cols[i*4+j], data=df, ax=axes[j])
