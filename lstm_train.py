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


class EarlyStoppingByFscore(Callback):
    def __init__(self,
                 filepath,
                 patience=0,
                 verbose=0):
        super(EarlyStoppingByFscore, self).__init__()

        self.filepath = filepath
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs={}):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
        self.best_epoch = -1

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_true = self.validation_data[1]

        predict = []
        true = []
        for p in val_predict:
            for pp in p:
                if pp[0] == 1:
                    predict.append(0)
                else:
                    predict.append(1)

        for t in val_true:
            for tt in t:
                if tt[0] == 1:
                    true.append(0)
                else:
                    true.append(1)

        try:
            _val_f1 = f1_score(true, predict)
            _val_re = recall_score(true, predict)
            _val_pr = precision_score(true, predict)
            print("Epoch {:05d}: f1_score is {}, recall is {}, precision is {}\n".format(
                epoch + 1, _val_f1, _val_re, _val_pr))
        except:
            print("Something exception happen.")
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print("best fscore is {}, on epoch {}\n".format(
                self.best, self.best_epoch))
            return

        if _val_f1 > self.best:
            print('Epoch %05d: f_score improved from %0.5f to %0.5f, saving model to %s\n'
                  % (epoch + 1, self.best, _val_f1, filepath))
            self.model.save(filepath, overwrite=True)
            self.best = _val_f1
            self.best_epoch = epoch + 1
            self.wait = 0

        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("best fscore is {}, on epoch {}\n".format(
                    self.best, self.best_epoch))

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping\n' % (self.stopped_epoch + 1))


for kpi_name in kpi_names:
    print("kpi_name: {}".format(kpi_name))
    kpi_df = train_interpolate_df[train_interpolate_df['KPI ID'] == kpi_name]
    kpi_size = len(kpi_df)
    value = kpi_df['value'].values.reshape(-1, 1)
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(value)
    label = kpi_df['label'].values.reshape(-1, 1)

    window_size = 120
    X = []
    y = []
    for i in range(kpi_size - window_size + 1):
        X.append(np.array(data[i:i + window_size]))

        new_arr = np.zeros((window_size, 2), dtype=np.int)
        temp = label[i:i + window_size]
        for i, t in enumerate(temp):
            new_arr[i][t] = 1

        y.append(np.array(new_arr))

    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    print(y.shape)

    true_index = []
    for i, yy in enumerate(y):
        if yy[:, 1].any() == 1:
            true_index.append(i)

    false_index = list(np.delete(list(range(X.shape[0])), true_index))
    true_index_len = len(true_index)
    false_index_len = len(false_index)

    np.random.shuffle(true_index)
    true_train_val_len = true_index_len // 2
    true_index_train = true_index[:true_train_val_len]
    true_index_val = true_index[true_train_val_len:]

    np.random.shuffle(false_index)
    false_train_val_len = false_index_len // 2
    false_index_train = false_index[:false_train_val_len]
    false_index_val = false_index[false_train_val_len:]

    ind_train = true_index_train + false_index_train
    ind_val = true_index_val + false_index_val

    print("train size is {}, true size is {}, false size is {}".
          format(len(ind_train), len(true_index_train), len(false_index_train)))
    print("validation size is {}, true size is {}, false size is {}".
          format(len(ind_val), len(true_index_val), len(false_index_val)))

    np.random.shuffle(ind_train)
    np.random.shuffle(ind_val)
    X_train = X[ind_train]
    y_train = y[ind_train]
    X_val = X[ind_val]
    y_val = y[ind_val]

    model = Sequential()
    model.add(LSTM(input_shape=(X.shape[1], X.shape[2]),
                   #batch_input_shape=(batch_size, input_length, input_dim),
                   units=128,
                   activation='tanh',
                   recurrent_activation='hard_sigmoid',
                   return_sequences=True,
                   # stateful=True,
                   dropout=0.2))

    model.add(LSTM(units=128,
                   activation='tanh',
                   recurrent_activation='hard_sigmoid',
                   return_sequences=True,
                   # stateful=True,
                   dropout=0.2))
    model.add(TimeDistributed(Dense(2, activation='softmax')))

    w_array = np.ones((2, 2))
    _round = 0
    while True:
        print("*" * 20 + "This is round {}".format(_round) + "*" * 20)
        print("Penalty on FN is {}".format(w_array[1, 0]))

        def loss(y_true, y_pred): return w_categorical_crossentropy(
            y_true, y_pred, weights=w_array)

        model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

        filepath = "model/" + kpi_name + ".h5"
        earlystop = EarlyStoppingByFscore(filepath, patience=50, verbose=1)
        callbacks_list = [earlystop]

        history = model.fit(X_train, y_train, batch_size=128, validation_data=(X_val, y_val),
                            callbacks=callbacks_list, verbose=1, epochs=500)

        if earlystop.best == 0:
            w_array[1, 0] += 0.5
            _round += 1
        else:
            print(earlystop.best)
            break
