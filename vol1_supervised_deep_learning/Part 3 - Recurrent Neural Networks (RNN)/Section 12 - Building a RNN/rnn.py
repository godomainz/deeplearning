import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

class Rnn:
    # Prepare a directory to store all the checkpoints.
    checkpoint_dir = 'ckpt'

    # Importing the Training Set
    dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
    training_set = dataset_train.iloc[:, 1:2].values

    # Feature scaling
    sc = MinMaxScaler(feature_range=(0, 1), copy=True)

    def __init__(self):
        print("tf.__version__ " + tf.__version__)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)


if __name__ == "__main__":
    my_rnn = Rnn()
