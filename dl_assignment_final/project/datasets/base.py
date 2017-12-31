import abc

import numpy


class Dataset(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractproperty
    def ishape(self):
        pass

    @abc.abstractproperty
    def osize(self):
        pass

    @abc.abstractmethod
    def batch(self, size=128):
        pass


class UniformFunctionDataset(Dataset):
    __metaclass__ = abc.ABCMeta

    def __init__(self, low=0, high=1):
        super(UniformFunctionDataset, self).__init__()
        self._low = low
        self._high = high

    def uniform(self, size):
        return numpy.random.rand(size, *self.ishape) * \
               (self._high - self._low) + self._low

    @abc.abstractmethod
    def function(self, X):
        pass

    def batch(self, size=128):
        X = self.uniform(size)
        return X, self.function(X)
