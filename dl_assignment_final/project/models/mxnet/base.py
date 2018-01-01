import abc

import mxnet

from project.models.base import Model
from mxnet.gluon import nn


class MXModel(Model):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, ishape, osize, hparam=None, path='./save',
                 dynamic=True):
        super(MXModel, self).__init__(name, hparam, path)

        self._context = mxnet.cpu()
        if dynamic:
            self._net = nn.Sequential()
            with self._net.name_scope():
                self._build(self._net, ishape, osize)
        else:
            raise NotImplementedError("Static graph is not implemented")

    @abc.abstractmethod
    def _build(self, net, ishape, osize):
        pass

    @staticmethod
    def get_activation(name):
        if name == 'relu':
            return 'relu'
        elif name is None:
            return None
        else:
            raise NotImplementedError('Activation \'{}\' is not implemented'
                                      .format(name))

    def save(self, epoch):
        self._net.save_params(self._path + '/checkpoints/{}/{}.params'
                              .format(epoch, self._name))

    def load(self, epoch):
        self._net.load_params(
            self._path + '/checkpoints/{}/{}.params'.format(epoch, self._name),
            ctx=self._context,
            allow_missing=False,
            ignore_extra=False
        )
