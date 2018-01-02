import abc

from project.models.base import Model


class TorchModel(Model):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, hparam=None, path='./save'):
        super(TorchModel, self).__init__(name, hparam, path)

    def save(self, epoch):
        pass

    def load(self, epoch):
        pass

    @abc.abstractmethod
    def _declare(self):
        pass

    @abc.abstractmethod
    def _forward(self):
        pass
