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
    replace_map = {column:{0:np.nan}}
    data.replace(replace_map, inplace=True)

    mean = data[column].mean()
    data[column] = data[column].fillna(0)
    replace_map = {column:{0:mean}}
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

def onehot_encode(train, test, cols):
    print("one-hot encoding:")
    from sklearn.preprocessing import OneHotEncoder
    for col in cols:
        print(f"\t{col}:", end='\t')

        train[col] = train[col].fillna("")
        test[col]  = test[col].fillna("")

        encoder = OneHotEncoder(sparse=False, categories='auto', handle_unknown='ignore')
        onehot_cols = encoder.fit_transform(train[col].to_numpy().reshape(-1,1))
        categories = encoder.categories_[0]
        col_names = [f"{col}_{cat}" for cat in categories]  # names of new onehot columns

        onehot_cols = pd.DataFrame(onehot_cols, columns=col_names)
        train = pd.concat([train.drop([col], axis='columns'), onehot_cols], axis='columns')

        onehot_cols = encoder.transform(test[col].to_numpy().reshape(-1,1))
        onehot_cols = pd.DataFrame(onehot_cols, columns=col_names)
        test = pd.concat([test.drop([col], axis='columns'), onehot_cols], axis='columns')
        
        print(str(len(categories)) + " categories")
    return train, test

def nominal_to_binary(train, test, cols):
    print("binary encoding:")
    #!pip install category_encoders
    from category_encoders import BinaryEncoder
    import math
    for col in cols:
        n_unique = len(train[col].unique())
        print('\t', col, end=" ")
        print('\t', n_unique, " unique values", " -> ", end='')
        print(math.ceil(math.log(n_unique, 2)), " columns")

        encoder = BinaryEncoder(verbose=1, cols=[col])
        train = encoder.fit_transform(train)
        test  = encoder.transform(test)

        i = 0
        while(f"col_{i}" in train.columns):
            rename_format = {f"col_{i}": f"{col}_col_{i}"}
            train = train.rename(columns=rename_format)
            test  = test.rename(columns=rename_format)
            i += 1
    return train, test

def preprocess_data(train, labels, test,
                    drop_cols, unique_cols, binary_cols, onehot_cols, scale_cols):
    # Drop irrelevant/redundant columns
    print(f"dropping: {drop_cols}")
    train = train.drop(drop_cols, axis='columns')
    test  = test.drop(drop_cols, axis='columns')

    # Onehot Encoding (train labels)
    labels = labels.pop('status_group').values
    labels = pd.get_dummies(labels)

    # Remove NaN's
    print("removing NaN's:")
    for col in binary_cols:
        print("\t", col)
        train[col] = train[col].fillna(False).astype('float64')
        test[col] = test[col].fillna(False).astype('float64')

    # Replace 0's with mean
    print("replacing 0's with mean:")
    for col in {"population", "amount_tsh", "construction_year"}:
        print("\t", col)
        train = clean_data_with_mean(train,col)
        test  = clean_data_with_mean(test,col)
    
    # Cyclic Encoding
    print("cyclic encoding:\n\tdate_recorded -> (doy_recorded_sin, doy_recorded_cos), year_recorded")
    train = date_to_cyclic(train)
    test = date_to_cyclic(test)

    # Normalization
    print("Normalization:")
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    scaler = StandardScaler()
    for col in scale_cols:
        print(f"\t{col} [{min(train[col]), max(train[col])}]", end=" -> ")
        train[col] = scaler.fit_transform(train[col].to_numpy().reshape(-1,1))
        test[col]  = scaler.transform(test[col].to_numpy().reshape(-1,1))
        print(f"[{min(train[col]), max(train[col])}]")

    # Onehot Encoding
    train, test = onehot_encode(train, test, onehot_cols)

    # Binary Encoding
    print("Binary Encoding:")
    train, test = nominal_to_binary(train, test, unique_cols)

    return train, labels, test

def load_train(path,path2, do_not_include, do_not_one_hot, clean_up, do_not_include_tent,
               do_not_include_temp, unique_cols):
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

    # Binary Encoding
    print("BINARY ENCODING:")
    import math
    #!pip install category_encoders
    from category_encoders import BinaryEncoder
    for col in unique_cols:
        n_unique = len(data[col].unique())
        print('\t', col, end=" ")
        print('\t', n_unique, " unique values", " -> ", end='')
        print(math.ceil(math.log(n_unique, 2)), " columns")

        encoder = BinaryEncoder(verbose=1, cols=[col])
        data = encoder.fit_transform(data)

        i = 0
        while(f"col_{i}" in data.columns):
            rename_format = {f"col_{i}": f"{col}_col_{i}"}
            data = data.rename(columns=rename_format)
            i += 1

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