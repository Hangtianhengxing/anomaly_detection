import numpy as np
import pandas as pd
import os
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from scipy.interpolate import interp1d

train_df = pd.read_csv('data/train.csv', dtype={'value': str})
train_df['value'] = train_df['value'].astype(float)
test_df = pd.read_csv('data/test.csv', dtype={'value': str})
test_df['value'] = test_df['value'].astype(float)

kpi_names = train_df['KPI ID'].values
kpi_names = np.unique(kpi_names)
train_interpolate_df = pd.DataFrame(
    columns=['timestamp', 'value', 'KPI ID', 'label', 'not_train'])

for kpi_name in kpi_names:
    kpi_name_df = train_df[train_df['KPI ID'] == kpi_name]
    kpi_timestamp = kpi_name_df['timestamp'].values
    kpi_value = kpi_name_df['value'].values
    kpi_label = kpi_name_df['label'].values
    interval = np.min(np.diff(kpi_timestamp))
    start_i = end_i = 0
    start_time = end_time = 0
    new_timestamp = []
    new_value = []
    new_label = []
    not_train = []

    for i, cur_time in enumerate(kpi_timestamp):
        if i == 0:
            start_time = cur_time
            start_i = i
        elif cur_time - start_time != interval and kpi_label[i] == kpi_label[start_i]:
            missing_number = int((cur_time - start_time) // interval - 1)
            x = [start_time, cur_time]
            start_value = kpi_value[start_i]
            cur_value = kpi_value[i]
            y = [start_value, cur_value]
            f_linear = interp1d(x, y)
            x_new = [start_time + interval * (n + 1)
                     for n in range(missing_number)]
            y_new = f_linear(x_new)
            new_timestamp.extend(x_new)
            new_value.extend(y_new)
            for n in range(missing_number):
                not_train.append(1)
                new_label.append(kpi_label[start_i])

        elif cur_time - start_time != interval:
            missing_number = int((cur_time - start_time) // interval - 1)
            x = [start_time, cur_time]
            start_value = kpi_value[start_i]
            cur_value = kpi_value[i]
            y = [start_value, cur_value]
            f_linear = interp1d(x, y)
            x_new = [start_time + interval * (n + 1)
                     for n in range(missing_number)]
            y_new = f_linear(x_new)
            new_timestamp.extend(x_new)
            new_value.extend(y_new)
            for n in range(missing_number):
                not_train.append(1)
                new_label.append(0)

        new_timestamp.append(cur_time)
        new_value.append(kpi_value[i])
        new_label.append(kpi_label[i])
        start_time = cur_time
        start_i = i
        not_train.append(0)

    assert len(new_timestamp) == len(new_value)
    assert len(new_timestamp) == len(not_train)
    new_id = [kpi_name] * len(new_timestamp)
    new_df = pd.DataFrame({'timestamp': new_timestamp, 'value': new_value, 'KPI ID': new_id, 'label': new_label,
                           'not_train': not_train})
    train_interpolate_df = pd.concat(
        [train_interpolate_df, new_df], ignore_index=True)

train_interpolate_df.to_csv('data/train_interpolate.csv', index=False)

test_interpolate_df = pd.DataFrame(
    columns=['timestamp', 'value', 'KPI ID', 'not_test'])

for kpi_name in kpi_names:
    kpi_name_df = test_df[test_df['KPI ID'] == kpi_name]
    kpi_timestamp = kpi_name_df['timestamp'].values
    kpi_value = kpi_name_df['value'].values
    interval = np.min(np.diff(kpi_timestamp))
    start_i = end_i = 0
    start_time = end_time = 0
    new_timestamp = []
    new_value = []
    not_test = []

    for i, cur_time in enumerate(kpi_timestamp):
        if i == 0:
            start_time = cur_time
        elif cur_time - start_time != interval:
            missing_number = int((cur_time - start_time) // interval - 1)
            x = [start_time, cur_time]
            start_value = kpi_value[start_i]
            cur_value = kpi_value[i]
            y = [start_value, cur_value]
            f_linear = interp1d(x, y)
            x_new = [start_time + interval * (n + 1)
                     for n in range(missing_number)]
            y_new = f_linear(x_new)
            new_timestamp.extend(x_new)
            new_value.extend(y_new)
            for n in range(missing_number):
                not_test.append(1)

        new_timestamp.append(cur_time)
        new_value.append(kpi_value[i])
        start_time = cur_time
        start_i = i
        not_test.append(0)

    assert len(new_timestamp) == len(new_value)
    assert len(new_timestamp) == len(not_test)
    new_id = [kpi_name] * len(new_timestamp)
    new_df = pd.DataFrame({'timestamp': new_timestamp,
                           'value': new_value, 'KPI ID': new_id, 'not_test': not_test})
    test_interpolate_df = pd.concat(
        [test_interpolate_df, new_df], ignore_index=True)

test_interpolate_df.to_csv('data/test_interpolate.csv', index=False)