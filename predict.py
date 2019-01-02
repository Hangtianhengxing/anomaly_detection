import numpy as np
import pandas as pd
import os
import math
import random
from collections import Counter
from itertools import product
from sklearn.utils import class_weight
from sklearn import preprocessing, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score, recall_score, precision_score
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten, TimeDistributed, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam, SGD
from keras import regularizers
from keras.layers.recurrent import LSTM
from keras import backend as K
import tensorflow as tf

train_interpolate_df = pd.read_csv('data/train_interpolate.csv')
test_interpolate_df = pd.read_csv('data/test_interpolate.csv')
test_df = pd.read_csv('data/test.csv')
kpi_names = train_interpolate_df['KPI ID'].values
kpi_names = np.unique(kpi_names)


def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, :, 0])
    y_pred_max = K.max(y_pred, axis=2)
    y_pred_max = K.reshape(
        y_pred_max, (K.shape(y_pred)[0], K.shape(y_pred)[1], 1))
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * K.cast(y_pred_max_mat, tf.float32)
                       [:, :, c_p] * K.cast(y_true, tf.float32)[:, :, c_t])
    return K.categorical_crossentropy(y_true, y_pred) * final_mask


label = np.zeros(len(test_df), dtype=np.int)
for kpi_name in kpi_names[1:2]:
    kpi_train_df = train_interpolate_df[train_interpolate_df['KPI ID'] == kpi_name]
    train_value = kpi_train_df['value'].values.reshape(-1, 1)
    scaler = preprocessing.StandardScaler()
    train_data = scaler.fit_transform(train_value)

    kpi_df = test_interpolate_df[test_interpolate_df['KPI ID'] == kpi_name]
    kpi_size = len(kpi_df)
    print("kpi_name: {}".format(kpi_name))
    test_value = kpi_df['value'].values.reshape(-1, 1)
    test_data = scaler.transform(test_value)

    kpi_test_df = test_df[test_df['KPI ID'] == kpi_name]
    not_test_index = kpi_df[kpi_df['not_test'] == 1].index - kpi_df.index[0]

    window_size = 120
    X = []
    for i in range(kpi_size - window_size + 1):
        X.append(np.array(test_data[i:i + window_size]))

    X_test = np.array(X)

    w_array = np.ones((2, 2))
    model = load_model('model/' + kpi_name + '.h5', custom_objects={
        'loss': lambda y_true, y_pred: w_categorical_crossentropy(y_true, y_pred, weights=w_array)})
    predict = model.predict(X_test)

    result = np.zeros((kpi_size, 2))
    for i, p in enumerate(predict):
        for j, pp in enumerate(p):
            result[i + j][0] += pp[0]
            result[i + j][1] += pp[1]

    final_result = np.zeros(kpi_size, dtype=np.int)
    for i in range(kpi_size):
        if result[i][1] >= result[i][0]:
            final_result[i] = 1

    y_test = np.delete(final_result, not_test_index)
    print(len(np.flatnonzero(y_test == 1)))
    assert len(y_test) == len(kpi_test_df)
    label[kpi_test_df.index[0]:kpi_test_df.index[-1] + 1] = y_test.copy()


label_df = pd.DataFrame(label, columns=['predict'])
test_df_no_value = test_df.drop(['value'], axis=1)
result_df = test_df_no_value.join(label_df)
pd.DataFrame(result_df).to_csv('result.csv', index=False)
