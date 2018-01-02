import pickle

import numpy
from sklearn.externals import joblib

from project.datasets.base import UniformFunctionDataset, Dataset


class Linear(UniformFunctionDataset):
    def __init__(self, k, b, low=0, high=1):
        super(Linear, self).__init__(low, high)
        self._k = k
        self._b = b

    def function(self, X):
        return self._k * X + self._b

    @property
    def ishape(self):
        return [1]

    @property
    def osize(self):
        return 1


class Gabor(UniformFunctionDataset):
    def __init__(self):
        super(Gabor, self).__init__()

    def function(self, X):
        return numpy.expand_dims(
            numpy.pi / 2 * numpy.exp(-2 * (X[:, 0] ** 2 + X[:, 1] ** 2)) *
            numpy.cos(2 * numpy.pi * (X[:, 0] + X[:, 1])), 1)

    @property
    def ishape(self):
        return [2]

    @property
    def osize(self):
        return 1


class PickleDataset(Dataset):
    def __init__(self, path):
        super(PickleDataset, self).__init__()
        with open(path, 'rb') as reader:
            self._X_train, self._y_train, self._X_eval, self._y_eval = \
                pickle.load(reader)

    @property
    def train(self):
        return self._X_train, self._y_train

    @property
    def eval(self):
        return self._X_eval, self._y_eval

    def batch(self, size=128):
        indexes = numpy.random.choice(len(self._X_train), size)
        return self._X_train[indexes], self._y_train[indexes]

    @property
    def osize(self):
        return self._y_train.shape[1:][0]

    @property
    def ishape(self):
        return self._X_train.shape[1:]


class JoblibDataset(Dataset):
    def __init__(self, path):
        super(JoblibDataset, self).__init__()
        with open(path, 'rb') as reader:
            self._X_train, self._y_train, self._X_eval, self._y_eval = \
                joblib.load(reader)

    @property
    def train(self):
        return self._X_train, self._y_train

    @property
    def eval(self):
        return self._X_eval, self._y_eval

    def batch(self, size=128):
        indexes = numpy.random.choice(len(self._X_train), size)
        return self._X_train[indexes], self._y_train[indexes]

    @property
    def osize(self):
        return self._y_train.shape[1:][0]

    @property
    def ishape(self):
        return self._X_train.shape[1:]