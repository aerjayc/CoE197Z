import pandas as pd
import numpy as np

from numpy import genfromtxt
from sklearn import preprocessing

import matplotlib.pyplot as plt
import string
from keras import backend as K

from sklearn.preprocessing import OneHotEncoder


#Given the dataframe and the column it replaces all zero nonavailable values with the mean
def clean_data_with_mean(data, column):
    replace_map = {column:{0:data[column].mean()}}
    # print(replace_map)
    data.replace(replace_map, inplace=True)
    return data

def date_to_cyclic(data):
    # For date_recorder, convert to three columns, 2 for cyclical day of year recorded and the year recorded
    # https://www.avanwyk.com/encoding-cyclical-features-for-deep-learning/
    data["date_recorded"] = pd.to_datetime(data["date_recorded"], format="%Y-%m-%d")
    data["day_of_year_recorded"] = data["date_recorded"].apply(lambda x: x.timetuple().tm_yday)

    data["year_recorded"] = data["date_recorded"].apply(lambda x: x.year)
    data["doy_recorded_sin"] = data["day_of_year_recorded"].apply(lambda x: np.sin(2 * np.pi * x/365.0))
    data["doy_recorded_cos"] = data["day_of_year_recorded"].apply(lambda x: np.cos(2 * np.pi * x/365.0))
    data = data.drop(["date_recorded", "day_of_year_recorded"], axis='columns')

    return data

def load_train(path,path2, do_not_include, do_not_one_hot, clean_up, do_not_include_tent, do_not_include_temp):
    ###Data Preprocessing
    data = pd.read_csv(path)
    labels = pd.read_csv(path2)

    # join data and labels
    data = data.join(labels.set_index('id'), on = 'id')

    # shuffle data
    data = data.sample(frac=1).reset_index(drop=True)
    
    # onehot encode the labels
    y = pd.concat([data['id'], data['status_group']], axis = 1)
    y = pd.get_dummies(y, columns=[y.columns[1]], prefix = [y.columns[1]])
    y = y.drop(columns = 'id').to_numpy()
    data = data.drop(columns = 'status_group')

    for col in clean_up:
        print("About to clean up")
        data = clean_data_with_mean(data,col)

    #Drop values not to be used
    data = data.drop(columns = do_not_include)
    data = data.drop(columns = 'id')
    data = data.drop(columns = do_not_include_tent)
    data = data.drop(columns = do_not_include_temp)

    data = date_to_cyclic(data)

    #Turn the rest into one hot
    print("one-hot encoding:")
    encoder = OneHotEncoder(sparse=False, categories='auto', handle_unknown='ignore')
    for col in data.columns:
        if col in ['public_meeting', 'permit']:
            data[col] = data[col].fillna(False)
        if col not in do_not_one_hot:
            print(f"\t{col}:", end='\t')

            data[col] = data[col].fillna("")

            onehot_cols = encoder.fit_transform(data[col].to_numpy().reshape(-1,1))
            categories = encoder.categories_[0]
            col_names = [f"{col}_{cat}" for cat in categories]  # names of new onehot columns

            onehot_cols = pd.DataFrame(onehot_cols, columns=col_names)
            data = pd.concat([data.drop([col], axis='columns'), onehot_cols], axis='columns')

            print(str(len(categories)) + " categories")

    train_col = data.columns

    x = data.to_numpy()

    #Normalize
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)

    try:
        del data
    except:
        pass

    return x, train_col, y



def load_x_test(path, train_col, do_not_include, do_not_one_hot, clean_up, do_not_include_tent, do_not_include_temp):
    ###Data Preprocessing
    data = pd.read_csv(path)
    
    for col in clean_up:
        data = clean_data_with_mean(data, col)

    data_id = data['id']
    #Drop values ot to be used
    data = data.drop(columns = do_not_include)
    data = data.drop(columns = 'id')
    
    pd.options.mode.chained_assignment = None
    for i in range(len(data['date_recorded'])):
        data['date_recorded'][i] = int(data['date_recorded'][i].replace("-","")[2:6])
        
    data = data.drop(columns = do_not_include_tent)
    data = data.drop(columns = do_not_include_temp)

    #Turn the rest into one hot
    for col in data.columns:
        # print("Before: ",len(data.columns),w)
        if col not in do_not_one_hot:
            prev = len(data.columns)
            data = pd.get_dummies(data, columns=[col], prefix = [col],dummy_na = True)
            now = len(data.columns)
            print("Expanded",col,"Change",prev,now)

    test_col = data.columns
    
    for i in range(len(test_col)):
        if test_col[i] not in train_col:
            data = data.drop(columns = test_col[i])
            print("Dropped",test_col[i])
    # print(data.columns,"InBetween")   
    data_ins = [0]*(data.shape[0])
    for i in range(len(train_col)):
        if train_col[i] not in test_col:
            index_name = train_col[i]
            index = i
            data.insert(index,index_name,data_ins)
            print("Included",train_col[i])
    
    if(train_col == data.columns).all():
        # print("Similar_Columns")
        pass
    x = data.to_numpy()
    # print("HELLO")  
    #Normalize
    min_max_scaler = preprocessing.MinMaxScaler()

    x = min_max_scaler.fit_transform(x)

    try:
        del data
    except:
        pass
    return x, test_col,data_id


# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))