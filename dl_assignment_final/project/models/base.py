import abc
import json
import os

from project.hparam import get_hparam, HParam


class Model(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, hparam=None, path='./save'):
        self._name = name
        if not hparam:
            self._hparam = self.default_hparam()
        else:
            assert isinstance(hparam, HParam)
            self._hparam = hparam
        self._path = os.path.realpath(path + '/' + self._name)
        if not os.path.exists(self._path):
            os.makedirs(self._path)
        if not os.path.exists(self._path + '/checkpoints'):
            os.makedirs(self._path + '/checkpoints')
        with open(self._path + '/hparam.json', 'w') as writer:
            json.dump(self._hparam, writer)

    @staticmethod
    def default_hparam():
        return get_hparam(seed=0)

    @abc.abstractmethod
    def save(self, epoch):
        pass

    @abc.abstractmethod
    def load(self, epoch):
        pass

    @abc.abstractmethod
    def train(self, X, y):
        pass

    @abc.abstractmethod
    def loss(self, X, y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass
