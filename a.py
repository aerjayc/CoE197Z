###Reyes, Marcus
###CoE 197Z Project 1
###Kaggle-https://www.kaggle.com/c/cat-in-the-dat

import pandas as pd
import keras
import numpy as np
from numpy import genfromtxt

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import adam

from sklearn import preprocessing


###Data preprocessing
data = pd.read_csv("train.csv")
#For now ignore the data you don't know how to handle
#drop = ['id', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
# drop = ['id', 'nom_7','nom_8', 'nom_9']
drop = ['id', 'nom_9']
data = data.drop(columns = drop)

#Categorical to one_hot
#https://www.datacamp.com/community/tutorials/categorical-data#encoding
one_hot = ['bin_3', 'bin_4','nom_0','nom_1','nom_2','nom_3','nom_4','ord_1', 'ord_2', 'ord_3', 'ord_4','ord_5','day','month', 'nom_5']

#Categorical to labelled
labelled = ['nom_6','nom_7', 'nom_8']

for i,w in enumerate(one_hot):
   data = pd.get_dummies(data, columns=[w], prefix = [w])
print(data.values.shape)

for i,w in enumerate(labelled):
    labels = data[w].astype('category').cat.categories.tolist()

    replace_map_comp = {w: {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

    data.replace(replace_map_comp, inplace=True)
    
    del labels, replace_map_comp
    
    print(data[w])

y_train = data['target'].to_numpy()
# print(y_train.shape)
y_train = keras.utils.to_categorical(y_train, 2)

data = data.drop(columns = ['target'])

x = data.to_numpy()

###Normalize data to large to be one-hot-encoded
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

# print(x[4,:])
# print(x[7,:])
x_train = x[:240000,:]
x_pretest = x[240000:,:]

y_pretest = y_train[240000:,:]
y_train = y_train[:240000,:]

###Model

hidden = 2048
dropout = 0.25
(trash, input_dim) = x.shape
model = Sequential()

model.add(Dense(hidden, input_dim = input_dim))
model.add(Dropout(dropout))
model.add(Activation('relu'))

model.add(Dense(hidden,input_dim = hidden))
model.add(Dropout(dropout))
model.add(Activation('relu'))

model.add(Dense(hidden,input_dim = hidden))
model.add(Dropout(dropout))
model.add(Activation('relu'))

model.add(Dense(2,input_dim = hidden))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

###To keep track of validation error
for i in range(7):

    model.fit(x_train, y_train, epochs = 2, batch_size = 4096*8)

    score = model.evaluate(x_pretest, y_pretest, batch_size = 512)
    print("\nTest accuacy: %.1f%%" % (100.0 * score[1]))


###Testing
try:
    del data
except:
    pass
 
data = pd.read_csv("test.csv")
data = data.drop(columns = drop)


for i,w in enumerate(one_hot):
   data = pd.get_dummies(data, columns=[w], prefix = [w])


for i,w in enumerate(labelled):
    labels = data[w].astype('category').cat.categories.tolist()

    replace_map_comp = {w: {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

    data.replace(replace_map_comp, inplace=True)
    
    del labels, replace_map_comp
    
    print(data[w])



x_test = data.to_numpy()

min_max_scaler = preprocessing.MinMaxScaler()
x_test = min_max_scaler.fit_transform(x_test)
 
# print("X_testshape",x_test.shape)
y_test = model.predict(x_test)



###Formatting into csv submittable
id = np.arange(start = 300000, stop = 500000)
id = np.transpose(id)
id = id.reshape(200000,1)
y_temp = y_test[:,1].reshape(200000,1)
y_pred = np.concatenate((id, y_temp), axis = 1)
print(id.shape)
print(y_test[:,0].shape)
print(y_pred.shape)
presubmission = pd.DataFrame(y_pred)

presubmission.iloc[:,0] = presubmission.iloc[:,0].astype(int)
presubmission.iloc[:,1] = presubmission.iloc[:,1].astype(float)


presubmission.to_csv("submission.csv",header = ["id","target"],index = False)