import abc

import mxnet
from mxnet import autograd

from project.hparam import get_hparam
from project.models.base import Model


class _Loss(mxnet.gluon.loss.Loss):
    def __init__(self, name, weight=None, batch_axis=0):
        super(_Loss, self).__init__(weight, batch_axis)
        self._name = name

    def hybrid_forward(self, F, answer, output):
        if self._name == 'mean_squared_error':
            return F.mean(F.sum(F.square(answer - output), axis=1),
                          axis=self._batch_axis)
        elif self._name == 'multilabel_sigmoid_cross_entropy':
            output = F.sigmoid(output)
            return -F.mean(F.sum(
                answer * F.log(output + 1e-12) +
                (1 - answer) * F.log(1 - output + 1e-12), axis=1),
                axis=self._batch_axis)
        else:
            raise NotImplementedError('Loss \'{}\' is not implemented'
                                      .format(self._name))


class MXModel(Model):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, hparam=None, path='./save'):
        super(MXModel, self).__init__(name, hparam, path)
        self._context = mxnet.cpu()
        self._net = mxnet.gluon.nn.HybridLambda(self._forward)
        with self._net.name_scope():
            for block in self._declare():
                self._net.register_child(block)
        self._net.hybridize()
        self._loss = _Loss(self._hparam.loss)
        self._net.collect_params().initialize(mxnet.init.Normal(sigma=0.01),
                                              ctx=self._context)
        self._trainer = mxnet.gluon.Trainer(
            self._net.collect_params(),
            self._hparam.optimizer.type,
            self._hparam.optimizer.params)

    def save(self, epoch):
        self._net.save_params(self._path + '/checkpoints/{}/params.model'
                              .format(epoch))

    def load(self, epoch):
        self._net.load_params(self._path + '/checkpoints/{}/params.model'
                              .format(epoch),
                              ctx=self._context,
                              allow_missing=False,
                              ignore_extra=False)

    @abc.abstractmethod
    def _forward(self, F, input):
        pass

    @abc.abstractmethod
    def _declare(self):
        pass

    def train(self, X, y):
        with autograd.record():
            X = mxnet.nd.array(X, ctx=self._context)
            y = mxnet.nd.array(y, ctx=self._context)
            output = self._net(X)
            loss = self._loss(y, output)
        loss.backward()
        self._trainer.step(X.shape[0])

    def loss(self, X, y):
        X = mxnet.nd.array(X, ctx=self._context)
        y = mxnet.nd.array(y, ctx=self._context)
        output = self._net(X)
        return self._loss(y, output).asscalar()

    def predict(self, X):
        X = mxnet.nd.array(X, ctx=self._context)
        return self._net(X).asnumpy()

    @staticmethod
    def default_hparam():
        result = get_hparam()
        result.update(MXModel.__bases__[0].default_hparam())
        return result
