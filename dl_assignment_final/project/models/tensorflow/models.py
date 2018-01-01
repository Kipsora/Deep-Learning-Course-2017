import functools
import operator

import tensorflow as tf

from project.hparam import get_hparam
from project.models.tensorflow.base import TFModel


class DeepNN(TFModel):
    def __init__(self, name, ishape, osize, hparam=None, path='./save'):
        super(DeepNN, self).__init__(name, ishape, osize, hparam, path)

    def loss(self, X, y):
        return self._session.run(self._loss, feed_dict={
            self._training: False,
            self._input: X,
            self._answer: y
        })

    def train(self, X, y):
        self._session.run(self._train_op, feed_dict={
            self._training: True,
            self._input: X,
            self._answer: y
        })

    def predict(self, X):
        return self._session.run(self._output, feed_dict={
            self._training: False,
            self._input: X
        })

    def _build(self, ishape, osize):
        self._training = tf.placeholder(tf.bool)
        self._input = tf.placeholder(tf.float32, shape=[None] + list(ishape))
        self._answer = tf.placeholder(tf.float32, shape=[None, osize])

        total = functools.reduce(operator.mul, ishape, 1)
        hidden = tf.reshape(self._input, shape=[-1, total])

        for i, layer in enumerate(self._hparam.layers):
            with tf.variable_scope('layer_' + str(i)):
                if layer.args:
                    hidden = tf.layers.dense(
                        inputs=hidden,
                        units=layer.units,
                        activation=self.get_activation(layer.activation),
                        **layer.args
                    )
                else:
                    hidden = tf.layers.dense(
                        inputs=hidden,
                        units=layer.units,
                        activation=self.get_activation(layer.activation)
                    )
                if layer.dropout:
                    hidden = tf.layers.dropout(
                        inputs=hidden,
                        training=self._training,
                        **layer.dropout
                    )
                if layer.batchnorm:
                    hidden = tf.layers.batch_normalization(
                        inputs=hidden,
                        training=self._training
                                 ** layer.batchnorm
                    )
        self._output = hidden
        self._loss = tf.reduce_mean(self.get_loss(self._hparam.loss,
                                                  self._answer, self._output))
        self._train_op = self.get_optimizer(
            self._hparam.optimizer).minimize(self._loss)

    @staticmethod
    def default_hparam():
        result = get_hparam(
            layers=[
                get_hparam(
                    units=20,
                    activation='leaky_relu',
                    batchnorm=get_hparam()
                ),
                get_hparam(
                    units=20,
                    activation='leaky_relu',
                    batchnorm=get_hparam()
                )
            ],
            loss='mean_squared_error',
            reg=get_hparam(
                type='L2',
                value=0.01
            ),
            optimizer=get_hparam(
                type='RMSProp',
                params=get_hparam(
                    learning_rate=0.001
                )
            )
        )
        result.update(TFModel.__bases__[0].default_hparam())
        return result
