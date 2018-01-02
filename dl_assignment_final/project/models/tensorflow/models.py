from __future__ import print_function
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
        with tf.variable_scope('network'):
            self._training = tf.placeholder(tf.bool)
            self._input = tf.placeholder(tf.float32, shape=[None] + list(ishape))
            self._answer = tf.placeholder(tf.float32, shape=[None, osize])
            batch_size = tf.shape(self._input)[0]

            hidden = self._input
            minor_loss = 0
            for i, config in enumerate(self._hparam.layers):
                with tf.variable_scope('layer_' + str(i)):
                    if config.batchnorm:
                        layer = tf.layers.BatchNormalization(
                            **config.batchnorm
                        )
                        hidden = layer(hidden, training=self._training)
                        if self._hparam.reg:
                            if self._hparam.reg.type == 'L2':
                                minor_loss += tf.reduce_mean(
                                    tf.square(layer.weights))
                            else:
                                raise NotImplementedError(
                                    '\'{}\' Regularization is not implemented')
                    if config.type == 'dense':
                        units = functools.reduce(
                            operator.mul, map(int, hidden.shape[1:]), 1)
                        hidden = tf.reshape(hidden, shape=[batch_size, units])
                        if config.args:
                            layer = tf.layers.Dense(
                                units=config.units,
                                activation=self.get_activation(
                                    config.activation),
                                **config.args
                            )
                        else:
                            layer = tf.layers.Dense(
                                units=config.units,
                                activation=self.get_activation(
                                    config.activation)
                            )
                        hidden = layer(hidden)
                        if self._hparam.reg:
                            if self._hparam.reg.type == 'L2':
                                minor_loss += tf.reduce_mean(
                                    tf.square(layer.kernel))
                            else:
                                raise NotImplementedError(
                                    '\'{}\' Regularization is'
                                    ' not implemented')
                    elif config.type == 'conv1d':
                        if config.args:
                            layer = tf.layers.Conv1D(
                                filters=config.filters,
                                kernel_size=config.kernel_size,
                                activation=self.get_activation(
                                    config.activation),
                                **config.args
                            )
                        else:
                            layer = tf.layers.Conv1D(
                                filters=config.filters,
                                kernel_size=config.kernel_size,
                                activation=self.get_activation(
                                    config.activation)
                            )
                        hidden = layer(hidden)
                        if self._hparam.reg:
                            if self._hparam.reg.type == 'L2':
                                minor_loss += tf.reduce_mean(
                                        tf.square(layer.kernel))
                            else:
                                raise NotImplementedError('\'{}\' Regularization is'
                                                          ' not implemented')
                    elif config.type == 'maxpool1d':
                        if config.args:
                            layer = tf.layers.MaxPooling1D(
                                pool_size=config.pool_size,
                                strides=config.strides,
                                **config.args
                            )
                        else:
                            layer = tf.layers.MaxPooling1D(
                                pool_size=config.pool_size,
                                strides=config.strides
                            )
                        hidden = layer(hidden)
                    else:
                        raise NotImplementedError('\'{}\' layer is not implemen'
                                                  'ted'.format(config.type))
                    if config.dropout:
                        hidden = tf.layers.dropout(
                            inputs=hidden,
                            training=self._training,
                            **config.dropout
                        )
            self._scores = hidden
            self._output, major_loss = self.get_loss(self._hparam.loss,
                                                     self._answer, self._scores)

            if self._hparam.reg:
                self._loss = major_loss + minor_loss * self._hparam.reg.value
            else:
                self._loss = major_loss

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
            reg=get_hparam(
                type='L2',
                value=0.001
            ),
            loss='mean_squared_error',
            optimizer=get_hparam(
                type='RMSProp',
                params=get_hparam(
                    learning_rate=0.001
                )
            )
        )
        result.update(DeepNN.__bases__[0].default_hparam())
        return result
