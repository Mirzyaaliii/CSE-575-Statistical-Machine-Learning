# Importing Libraries
import os
import librosa
import librosa.display
import pathlib
import numpy as np
import matplotlib.pyplot as plt


# Helper Functions
def fetch_spectograms(path):
    X = []
    y = []
    data_dir = os.listdir(path)
    for folder in data_dir:
        dir_path = path / folder
        data_items = os.listdir(dir_path)
        for num, item in enumerate(data_items):
            item_path = dir_path / item
            img = plt.imread(item_path)[:,:,:3]
            X.append(img)
            y.append(folder)
            print(f"\rLoaded {num+1} files in {folder}",end="")
        print("")
    return np.array(X),np.array(y)


def generate_spectograms(path):
    X = []
    y = []
    data_dir = os.listdir(path)
    for folder in data_dir:
        dir_path = path / folder
        data_items = os.listdir(dir_path)
        for num, item in enumerate(data_items):
            item_path = dir_path / item
            try:
                m,sr = librosa.load(item_path)
                data = librosa.feature.melspectrogram(y=m, sr=sr)
                data_db = librosa.power_to_db(data, ref=np.max)
                img = librosa.display.specshow(data_db, sr=sr, fmax=8000)
            except Exception as e:
                print(f"\nError in file: {item_path}, {e}")
            X.append(img)
            y.append(folder)
            print(f"\rLoaded {num+1} files in {folder}",end="")
        print("")
    return X,y


if __name__ == "__main__":
    # Defining Parameters
    DATA_PATH = pathlib.Path("D:\\Datasets\\GTZAN\\Data\\images_original")
    # Loading Data
    X,y = fetch_spectograms(DATA_PATH)
    
    np.save("../X.npy",X)
    np.save("../y.npy",y)
