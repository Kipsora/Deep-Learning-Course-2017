import abc
import tensorflow as tf

from project.models.base import Model
from project.hparam import get_hparam


class TFModel(Model):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, ishape, osize, hparam=None, path='./save'):
        super(TFModel, self).__init__(name, hparam, path)

        with tf.variable_scope(self._name):
            self._build(ishape, osize)

        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True
        self._session = tf.Session(config)
        self._session.run(tf.global_variables_initializer())
        self._session.run(tf.local_variables_initializer())

    @staticmethod
    def get_activation(name):
        if name == 'leaky_relu':
            return tf.nn.leaky_relu
        elif name == 'relu':
            return tf.nn.relu
        elif name == 'tanh':
            return tf.nn.tanh
        elif name == 'sigmoid':
            return tf.nn.sigmoid
        elif name is None:
            return None
        else:
            raise NotImplementedError('Activation \'{}\' is not implemented'
                                      .format(name))

    @staticmethod
    def get_loss(name, answer, output):
        if name == 'mean_squared_error':
            return tf.reduce_sum(tf.square(answer - output), axis=-1)
        elif name == 'cross_entropy':
            return tf.nn.softmax_cross_entropy_with_logits(
                labels=answer, logits=output)
        else:
            raise NotImplementedError('Loss \'{}\' is not implemented'
                                      .format(name))

    @staticmethod
    def get_optimizer(hparam):
        if hparam.type == 'Adam':
            return tf.train.AdamOptimizer(**hparam.params)
        elif hparam.type == 'RMSProp':
            return tf.train.RMSPropOptimizer(**hparam.params)
        else:
            raise NotImplementedError('Optimizer \'{}\' is not implemented'
                                      .format(hparam.type))

    @staticmethod
    def default_hparam():
        result = get_hparam()
        result.update(TFModel.__bases__[0].default_hparam())
        return result

    @abc.abstractmethod
    def _build(self, ishape, osize):
        pass

    def save(self, epoch):
        saver = tf.train.Saver(tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self._name))
        saver.save(
            self._session,
            self._path + '/checkpoints/{}/{}'.format(epoch, self._name))

    def load(self, epoch):
        saver = tf.train.Saver(tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self._name))
        saver.restore(
            self._session,
            self._path + '/checkpoints/{}/{}'.format(epoch, self._name))
