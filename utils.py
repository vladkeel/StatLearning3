import requests
import pandas as pd

from sklearn.datasets import fetch_mldata


def load_mnist():
    try:
        mnist = fetch_mldata('MNIST original')

        data_df = pd.DataFrame(mnist['data'])
        label_df = pd.DataFrame(mnist['target'])
        data = pd.concat([data_df, label_df], axis=1)
        data = data.sample(axis=0, n=8000)
        data_df = data.iloc[:, :-1]
        label_df = data.iloc[:, -1]
        return data_df, label_df
    except requests.exceptions.RequestException:
        print('HTTP exception, check you connection and try again')
