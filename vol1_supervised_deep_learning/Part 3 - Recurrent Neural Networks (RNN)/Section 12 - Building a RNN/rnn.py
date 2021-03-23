import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Rnn:

    # Importing the Training Set
    dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
    training_set = dataset_train.iloc[:, 1:2].values

    def __init__(self):
        print("RNN")


if __name__ == "__main__":
    my_rnn = Rnn()
