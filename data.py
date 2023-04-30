import os
import io
import tempfile
import numpy as np
import pandas as pd

from PIL import Image
from urllib.request import urlretrieve
from zipfile import ZipFile

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def encode_target(y):
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)
    labels = encoder.categories_[0]
    return y, labels


def split_data(X, y, train_split=0.6, val_split=0.2, random_state=0):
    n_samples = len(y)
    indices = np.arange(n_samples)

    indices_train, indices_test, y_train, y_test = train_test_split(indices,
                                                                    y,
                                                                    train_size=round(train_split * n_samples),
                                                                    stratify=y,
                                                                    random_state=random_state)

    indices_val, indices_test, y_val, y_test = train_test_split(indices_test,
                                                                y_test,
                                                                train_size=round(val_split * n_samples),
                                                                stratify=y_test,
                                                                random_state=random_state)

    return X[indices_train], X[indices_val], X[indices_test], y_train, y_val, y_test


def get_data_ANN(random_state=0):
    df = pd.read_csv(
        'https://raw.githubusercontent.com/prat-man/CSE-575-Statistical-Machine-Learning/main/data/features_30_sec.csv')

    df.drop(['filename', 'length'], axis=1, inplace=True)

    X = df.loc[:, df.columns != 'label'].to_numpy()
    y = df.loc[:, df.columns == 'label'].to_numpy()
    y, labels = encode_target(y)

    # split into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_state=random_state)

    # standard scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, labels


def get_data_CNN(cropped=False, random_state=0):
    zip_path = tempfile.gettempdir() + '/spectrogram.zip'

    if not os.path.isfile(zip_path):
        urlretrieve(
            'https://raw.githubusercontent.com/prat-man/CSE-575-Statistical-Machine-Learning/main/data/spectrogram.zip',
            zip_path)

    X = []
    y = []

    with ZipFile(zip_path, 'r') as zip:
        for name in zip.namelist():
            genre = name.split('/')[0]
            # file_name = name.split('/')[1]

            image_data = zip.read(name)
            image = Image.open(io.BytesIO(image_data))

            if cropped:
                np_image = np.array(image)[35:253, 54:390, :3]
            else:
                np_image = np.array(image)[:, :, :3]

            X.append(np_image)
            y.append(genre)

    X = np.array(X)
    y = np.array(y, ndmin=2).T
    y, labels = encode_target(y)

    # split into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test, labels


def get_data_LSTM(random_state=0):
    df = pd.read_csv(
        'https://raw.githubusercontent.com/prat-man/CSE-575-Statistical-Machine-Learning/main/data/features_3_sec.csv')

    df['filename'] = df['filename'].str[:-6]
    df.drop('length', axis=1, inplace=True)

    groups = df.groupby('filename')

    X = []
    y = []
    for group_name, group in groups:
        X.append(group.loc[:, ~group.columns.isin(['filename', 'label'])].to_numpy())
        y.append(group.iloc[0]['label'])

    max_0 = max([x.shape[0] for x in X])
    for i, x in enumerate(X):
        if max_0 != x.shape[0]:
            padding = np.zeros((max_0 - x.shape[0], x.shape[1]))
            X[i] = np.vstack((x, padding))

    X = np.stack(X)
    y = np.array(y)[:, np.newaxis]
    y, labels = encode_target(y)

    # split into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_state=random_state)

    # standard scaling
    scaler = StandardScaler()

    train_shape = X_train.shape
    train_new_shape = X_train.shape[0] * X_train.shape[1], X_train.shape[2]
    X_train = scaler.fit_transform(X_train.reshape(train_new_shape)).reshape(train_shape)

    val_shape = X_val.shape
    val_new_shape = X_val.shape[0] * X_val.shape[1], X_val.shape[2]
    X_val = scaler.transform(X_val.reshape(val_new_shape)).reshape(val_shape)

    test_shape = X_test.shape
    test_new_shape = X_test.shape[0] * X_test.shape[1], X_test.shape[2]
    X_test = scaler.transform(X_test.reshape(test_new_shape)).reshape(test_shape)

    return X_train, X_val, X_test, y_train, y_val, y_test, labels


def get_data(model, random_state=0):
    match model:
        case 'ANN':
            return get_data_ANN(random_state=random_state)

        case 'CNN':
            return get_data_CNN(random_state=random_state)

        case 'LSTM':
            return get_data_LSTM(random_state=random_state)

        case _:
            return None
