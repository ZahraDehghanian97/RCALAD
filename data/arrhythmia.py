import logging
import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

final_dataset = {}
logger = logging.getLogger(__name__)


def get_train(label=0, scale=False, *args):
    """Get training dataset for Thyroid dataset"""
    return _get_adapted_dataset("train", scale)


def get_test(label=0, scale=False, *args):
    """Get testing dataset for Thyroid dataset"""
    return _get_adapted_dataset("test", scale)


def get_valid(label=0, scale=False, *args):
    """Get validation dataset for Thyroid dataset"""
    global final_dataset
    x_valid, x_test, \
    y_valid, y_test = train_test_split(final_dataset['x_test'],
                                       final_dataset['y_test'],
                                       test_size=0.5,
                                       random_state=42)
    dataset = {'x_train': final_dataset['x_train'], 'y_train': final_dataset['y_train'],
               'x_valid': x_valid.astype(np.float32), 'y_valid': y_valid.astype(np.float32),
               'x_test': x_test.astype(np.float32), 'y_test': y_test.astype(np.float32)}
    final_dataset = dataset
    # print("Size of split validation :", dataset['x_valid'].shape[0])
    return final_dataset['x_valid'], final_dataset['y_valid']


def get_shape_input():
    """Get shape of the dataset for Thyroid dataset"""
    return (None, 274)


def get_shape_input_flatten():
    """Get shape of the dataset for Thyroid dataset"""
    return (None, 274)


def get_shape_label():
    """Get shape of the labels in Thyroid dataset"""
    return (None,)


def get_anomalous_proportion():
    return 0.15


def _get_dataset(scale):
    """ Gets the basic dataset
    Returns :
            dataset (dict): containing the data
                dataset['x_train'] (np.array): training images shape
                (?, 120)
                dataset['y_train'] (np.array): training labels shape
                (?,)
                dataset['x_test'] (np.array): testing images shape
                (?, 120)
                dataset['y_test'] (np.array): testing labels shape
                (?,)
    """
    global final_dataset
    if not bool(final_dataset):
        data = scipy.io.loadmat("D:/univesity/foqelisans/final_project/code/Adversarially-Learned-Anomaly-Detection_dxxzz/data/arrhythmia.mat")
        # data = scipy.io.loadmat("/content/drive/MyDrive/colab/ALAD/arrhythmia.mat")

        full_x_data = data["X"]
        full_y_data = data['y']

        x_train, x_test, \
        y_train, y_test = train_test_split(full_x_data,
                                           full_y_data,
                                           test_size=0.5,
                                           random_state=42)

        y_train = y_train.flatten().astype(int)
        y_test = y_test.flatten().astype(int)

        inliers = x_train[y_train == 0], y_train[y_train == 0]
        x_train, y_train = inliers

        if scale:
            print("Scaling dataset")
            scaler = MinMaxScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
        dataset = {'x_train': x_train.astype(np.float32), 'y_train': y_train.astype(np.float32),
                   'x_test': x_test.astype(np.float32), 'y_test': y_test.astype(np.float32)}
        final_dataset = dataset
    # print(final_dataset.keys())
    return final_dataset


def _get_adapted_dataset(split, scale):
    """ Gets the adapted dataset for the experiments

    Args :
            split (str): train or test
    Returns :
            (tuple): <training, testing> images and labels
    """
    # print("_get_adapted",scale)
    dataset = _get_dataset(scale)
    key_img = 'x_' + split
    key_lbl = 'y_' + split
    print("Size of split", split, ":", dataset[key_lbl].shape[0])
    print("size of data : ",dataset[key_img].shape)
    return dataset[key_img], dataset[key_lbl]


def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)
