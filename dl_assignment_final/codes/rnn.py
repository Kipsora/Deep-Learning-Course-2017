from __future__ import print_function

import numpy
import tensorflow as tf
from sklearn.externals import joblib

from project.utils import get_data, tensorflow_gpu_config, metrics, set_seed, plot

tf.flags.DEFINE_integer('print_period', default_value=100, docstring='')
tf.flags.DEFINE_integer('n_epoch', default_value=2000, docstring='')
tf.flags.DEFINE_integer('load_from', default_value=None, docstring='')
tf.flags.DEFINE_integer('batch_size', default_value=64, docstring='')
tf.flags.DEFINE_float('clip_by_norm', default_value=5.0, docstring='')
tf.flags.DEFINE_float('learning_rate', default_value=0.0001, docstring='')
tf.flags.DEFINE_float('init_stddev', default_value=0.01, docstring='')
tf.flags.DEFINE_float('reg', default_value=0.01, docstring='')
tf.flags.DEFINE_float('train_noise', default_value=0.0, docstring='')
tf.flags.DEFINE_integer('train_copy', default_value=1, docstring='')
tf.flags.DEFINE_string('split', default_value='audioset/small/raw/test',
                       docstring='')
tf.flags.DEFINE_integer('seed', default_value=None, docstring='')


def dense(inputs, units, activation, kernel_initializer, kernel_regularizer):
    layer = tf.layers.Dense(units, activation=activation,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer)
    result = layer(inputs)
    return result, layer.kernel, layer.bias


def conv(inputs, kernel_size, activation, initializer, regularizer,
         pool_size=None):
    hidden = tf.layers.conv1d(inputs, filters=256,
                              kernel_size=kernel_size, activation=activation,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer)
    if pool_size is not None:
        hidden = tf.layers.max_pooling1d(hidden, pool_size=pool_size, strides=1)
    return hidden


def main(_):
    config = tf.flags.FLAGS
    print('seed', '=', set_seed(config.seed))

    with tf.variable_scope('network'):
        with tf.variable_scope('utility'):
            initializer = tf.truncated_normal_initializer(mean=0, stddev=0.01)
            regularizer = tf.contrib.layers.l2_regularizer(config.reg)

        with tf.variable_scope('inputs'):
            training = tf.placeholder(tf.bool)
            learning_rate = tf.placeholder(tf.float32)
            lengths = tf.placeholder(tf.int32, shape=[None])
            inputs = tf.placeholder(tf.float32, shape=[None, 10, 128])
            answers = tf.placeholder(tf.float32, shape=[None, 527])

        with tf.variable_scope('process'):
            hidden = inputs

            lstm_stacked = tf.contrib.rnn.MultiRNNCell(
                [tf.nn.rnn_cell.LSTMCell(2048, activation=tf.nn.tanh)
                 for _ in range(10)])
            _, final_state = tf.nn.dynamic_rnn(lstm_stacked, hidden,
                                               dtype=tf.float32)
            hidden0 = tf.concat(final_state[0], axis=1)
            hidden1 = tf.concat(final_state[1], axis=1)
            hidden = tf.concat((hidden0, hidden1), axis=1)

            hidden = tf.layers.batch_normalization(hidden)
            hidden, _, _ = dense(hidden, 1024, None, initializer, regularizer)
            hidden = tf.nn.leaky_relu(hidden)
            hidden = tf.layers.dropout(hidden, training=training)

        with tf.variable_scope('outputs'):
            scores, _, _ = dense(hidden, 527, None, initializer, regularizer)
            predicts = tf.nn.sigmoid(scores)

        with tf.variable_scope('loss'):
            cls_loss = tf.reduce_mean(
                tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=answers, logits=scores
                ), axis=-1))
            reg_loss = tf.losses.get_regularization_loss()
            loss = cls_loss + reg_loss

        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            if config.clip_by_norm:
                gradients = [None if gradient is None
                             else tf.clip_by_norm(gradient, config.clip_by_norm)
                             for gradient in gradients]
            train = optimizer.apply_gradients(zip(gradients, variables))

    if config.load_from:
        load_from = config.load_from
        # TODO: Load from models
    else:
        load_from = 0

    session = tf.Session(config=tensorflow_gpu_config())
    session.run(tf.global_variables_initializer())

    train_set, eval_set, test_set = get_data(
        path='audioset/small',
        split=config.split,
        train_noise=config.train_noise,
        train_copy=config.train_copy)
    print(train_set.shape)
    print(eval_set.shape)
    print(test_set.shape)

    plt_X, plt_train, plt_eval, plt_AUC, plt_AP = [], [], [], [], []
    for epoch in range(load_from + 1, load_from + config.n_epoch + 1):
        X_batch, y_batch, l_batch = train_set.batch(size=config.batch_size)
        _, train_loss, train_cls_loss, train_reg_loss = \
            session.run([train, loss, cls_loss, reg_loss], feed_dict={
                inputs: X_batch,
                answers: y_batch,
                lengths: l_batch,
                training: True,
                learning_rate: config.learning_rate
            })
        if epoch % config.print_period == 0 or epoch == load_from:
            eval_predicts = []
            eval_loss = []
            eval_reg_loss = []
            eval_cls_loss = []
            for i in range(0, len(eval_set.X), 500):
                current_loss, current_cls_loss, current_reg_loss, current_predicts = \
                    session.run((loss, cls_loss, reg_loss, predicts),
                                feed_dict={
                                    inputs: eval_set.X[i:i + 500],
                                    answers: eval_set.y[i:i + 500],
                                    lengths: eval_set.length[i:i + 500],
                                    training: False
                                })
                eval_predicts.append(current_predicts)
                eval_loss.append(current_loss)
                eval_reg_loss.append(current_reg_loss)
                eval_cls_loss.append(current_cls_loss)
            eval_predicts = numpy.concatenate(eval_predicts, axis=0)
            eval_loss = numpy.mean(numpy.asarray(eval_loss))
            eval_cls_loss = numpy.mean(numpy.asarray(eval_cls_loss))
            eval_reg_loss = numpy.mean(numpy.asarray(eval_reg_loss))

            auc, ap = metrics(eval_set.y, eval_predicts)
            print('epoch {}/{} train loss: {} eval: loss: {} auc: {} ap: {}'
                  .format(epoch, load_from + config.n_epoch,
                          (train_loss, train_cls_loss, train_reg_loss),
                          (eval_loss, eval_cls_loss, eval_reg_loss),
                          auc, ap))
            plt_X.append(epoch)
            plt_train.append(train_loss)
            plt_eval.append(eval_loss)
            plt_AUC.append(auc)
            plt_AP.append(ap)
    plot(((plt_X, plt_train), (plt_X, plt_eval)),
     legends=('Train', 'Eval'), log='rnn-loss.log', fig='rnn-loss.png')
    plot(((plt_X, plt_AUC),),
         legends=('Eval AUC',), log='rnn-AUC.log', fig='rnn-AUC.png')
    plot(((plt_X, plt_AP),),
        legends=('Eval AP',), log='rnn-AP.log', fig='rnn-AP.png')

    eval_predicts = []
    eval_loss = []
    eval_reg_loss = []
    eval_cls_loss = []
    for i in range(0, len(test_set.X), 500):
        current_loss, current_cls_loss, current_reg_loss, current_predicts = \
            session.run((loss, cls_loss, reg_loss, predicts), feed_dict={
                inputs: test_set.X[i:i + 500],
                answers: test_set.y[i:i + 500],
                lengths: test_set.length[i:i + 500],
                training: False
            })
        eval_predicts.append(current_predicts)
        eval_loss.append(current_loss)
        eval_reg_loss.append(current_reg_loss)
        eval_cls_loss.append(current_cls_loss)
    eval_predicts = numpy.concatenate(eval_predicts, axis=0)
    eval_loss = numpy.mean(numpy.asarray(eval_loss))
    eval_cls_loss = numpy.mean(numpy.asarray(eval_cls_loss))
    eval_reg_loss = numpy.mean(numpy.asarray(eval_reg_loss))
    auc, ap = metrics(test_set.y, eval_predicts)
    print('test: loss: {} auc: {} ap: {}'
          .format((eval_loss, eval_cls_loss, eval_reg_loss),
                  auc, ap))
    with open('rnn.predicts', 'wb') as writer:
        joblib.dump(eval_predicts, writer)


if __name__ == '__main__':
    tf.app.run()
