import io
import numpy as np
import os
from PIL import Image
import pandas as pd
from urllib.request import urlretrieve
from zipfile import ZipFile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


random_seed = 42


## Data preparation for CNN model
def data_spectrogram(zip_file_path, train_list, val_list, test_list, cropped=False):
  
  X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []

  with ZipFile('spectrogram.zip', 'r') as zip:

    for each_filename in train_list:
      genre, id = each_filename.split(".")[0], each_filename.split(".")[1]
      name = genre+"/"+genre+id+".png"
      image_data = zip.read(name)
      image = Image.open(io.BytesIO(image_data))
      if cropped:
        np_image = np.array(image)[35:253, 54:390, :3]
      np_image = np.array(image)[:, :, :3]
      X_train.append(np_image)
      y_train.append(genre)

    for each_filename in val_list:
      genre, id = each_filename.split(".")[0], each_filename.split(".")[1]
      name = genre+"/"+genre+id+".png"
      image_data = zip.read(name)
      image = Image.open(io.BytesIO(image_data))
      if cropped:
        np_image = np.array(image)[35:253, 54:390, :3]
      np_image = np.array(image)[:, :, :3]
      X_val.append(np_image)
      y_val.append(genre)

    for each_filename in test_list:
      genre, id = each_filename.split(".")[0], each_filename.split(".")[1]
      name = genre+"/"+genre+id+".png"
      image_data = zip.read(name)
      image = Image.open(io.BytesIO(image_data))
      if cropped:
        np_image = np.array(image)[35:253, 54:390, :3]
      np_image = np.array(image)[:, :, :3]
      X_test.append(np_image)
      y_test.append(genre)

  X_train = np.array(X_train)
  y_train = np.array(y_train, ndmin=2).T
  X_val = np.array(X_val)
  y_val = np.array(y_val, ndmin=2).T
  X_test = np.array(X_test)
  y_test = np.array(y_test, ndmin=2).T

  return X_train, y_train, X_val, y_val, X_test, y_test


## Data preparation for ANN model
def data_3_sec(sec_3_csv_path, train_list, val_list, test_list, encoder):

  sec_3_df = pd.read_csv(sec_3_csv_path)

  train_df = pd.DataFrame()
  val_df = pd.DataFrame()
  test_df = pd.DataFrame()

  for each_filename in train_list:
    genre, id = each_filename.split(".")[0], each_filename.split(".")[1]
    current_df = sec_3_df[(sec_3_df['filename'].str.contains(genre)) & (sec_3_df['filename'].str.contains(id))]
    train_df = pd.concat([train_df, current_df], axis=0)

  for each_filename in val_list:
    genre, id = each_filename.split(".")[0], each_filename.split(".")[1]
    current_df = sec_3_df[(sec_3_df['filename'].str.contains(genre)) & (sec_3_df['filename'].str.contains(id))]
    val_df = pd.concat([val_df, current_df], axis=0)

  for each_filename in test_list:
    genre, id = each_filename.split(".")[0], each_filename.split(".")[1]
    current_df = sec_3_df[(sec_3_df['filename'].str.contains(genre)) & (sec_3_df['filename'].str.contains(id))]
    test_df = pd.concat([test_df, current_df], axis=0)

  X_train = train_df.loc[:, train_df.columns != 'label']
  y_train = train_df.loc[:, train_df.columns == 'label']
  X_val = val_df.loc[:, val_df.columns != 'label']
  y_val = val_df.loc[:, val_df.columns == 'label']
  X_test = test_df.loc[:, test_df.columns != 'label']
  y_test = test_df.loc[:, test_df.columns == 'label']

  X_train = X_train.drop(["filename", "length"], axis=1)
  X_val = X_val.drop(["filename", "length"], axis=1)
  X_test = X_test.drop(["filename", "length"], axis=1)

  y_train = encoder.fit_transform(y_train)
  y_val = encoder.fit_transform(y_val)
  y_test = encoder.fit_transform(y_test)
  labels = encoder.categories_[0]

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_val = scaler.transform(X_val)
  X_test = scaler.transform(X_test)
  X_train, y_train, X_val, y_val, X_test, y_test  = X_train.to_numpy(), y_train.to_numpy(), X_val.to_numpy(), y_val.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

  return X_train, y_train, X_val, y_val, X_test, y_test, labels


## Main function for data preprocessing 
def data_preprocess(data_dir, training_type):

  sec_30_csv_path = os.path.join(data_dir, "features_30_sec.csv")
  sec_30_csv_df = pd.read_csv(sec_30_csv_path)
  sec_30_csv_df = sec_30_csv_df.drop(sec_30_csv_df[sec_30_csv_df['filename']=="jazz.00054.wav"].index)

  n_samples = sec_30_csv_df.shape[0]
  train_split = 0.7
  val_split = 0.1
  test_split = 0.2

  X = sec_30_csv_df.loc[:, sec_30_csv_df.columns != 'label']
  y = sec_30_csv_df.loc[:, sec_30_csv_df.columns == 'label']

  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=round(train_split*n_samples), stratify=y, random_state=random_seed)
  X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=round(val_split*n_samples), stratify=y_test, random_state=random_seed)

  encoder = OneHotEncoder(sparse_output=False)

  if training_type == "ANN_30":
    X_train = X_train.drop(["filename", "length"], axis=1)
    X_val = X_val.drop(["filename", "length"], axis=1)
    X_test = X_test.drop(["filename", "length"], axis=1)

    y_train = encoder.fit_transform(y_train)
    y_val = encoder.fit_transform(y_val)
    y_test = encoder.fit_transform(y_test)
    labels = encoder.categories_[0]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_train, y_train, X_val, y_val, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_val.to_numpy(), y_val.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    return X_train, y_train, X_val, y_val, X_test, y_test, labels

  train_list, val_list, test_list = X_train['filename'].tolist(), X_val['filename'].tolist(), X_test['filename'].tolist()

  if training_type == "ANN_3":
    sec_3_csv_path = os.path.join(data_dir, "features_3_sec.csv")
    X_train, y_train, X_val, y_val, X_test, y_test, labels = data_3_sec(sec_3_csv_path, train_list, val_list, test_list, encoder)
    return X_train, y_train, X_val, y_val, X_test, y_test, labels

  if "CNN" in training_type:

    zip_file_path = os.path.join(data_dir, "spectrogram.zip")
    urlretrieve(zip_file_path, "spectrogram.zip")

    if training_type == "CNN":
      X_train, y_train, X_val, y_val, X_test, y_test = data_spectrogram("spectrogram.zip", train_list, val_list, test_list, cropped=False)
      return X_train, y_train, X_val, y_val, X_test, y_test

    if training_type == "CNN_cropped":
      X_train, y_train, X_val, y_val, X_test, y_test = data_spectrogram("spectrogram.zip", train_list, val_list, test_list, cropped=True)
      return X_train, y_train, X_val, y_val, X_test, y_test
