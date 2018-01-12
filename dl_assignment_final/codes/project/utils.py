from __future__ import absolute_import

import gzip
import logging
import pickle
import sys
import os

import mxnet
import numpy
import tensorflow
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


class DataSet:
    def __init__(self, X, y, length):
        self.X = numpy.asarray(X, dtype=numpy.float32)
        self.y = numpy.asarray(y, dtype=numpy.int32)
        self.length = numpy.asarray(length, dtype=numpy.int32)
        indexes = numpy.arange(0, len(self.X))
        numpy.random.shuffle(indexes)
        self.X = self.X[indexes]
        self.y = self.y[indexes]
        self.length = self.length[indexes]

    def batch(self, size=128):
        indexes = numpy.random.choice(len(self.X), size)
        X = self.X[indexes]
        y = self.y[indexes]
        l = self.length[indexes]
        return X, y, l

    @property
    def shape(self):
        return self.X.shape, self.y.shape


def tensorflow_gpu_config():
    config = tensorflow.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    config.gpu_options.allow_growth = True
    return config


def get_logger(scope, log_path=None, level='INFO'):
    logger = logging.getLogger(scope)
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    if log_path:
        handlers.append(logging.FileHandler(filename=log_path))
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    logger.setLevel(level)
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def plot(data, log=None, fig=None, figsize=(8, 6),
         xlabel=None, ylabel=None, title=None, legends=None):
    import matplotlib
    if 'DISPLAY' not in os.environ:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    for (X, y) in data:
        plt.plot(X, y)
    if legends:
        plt.legend(legends)
    if title:
        plt.title(title)
    if fig:
        plt.savefig(fig)
    if log:
        if legends:
            with open(log, 'w') as writer:
                for (X, y), l in zip(data, legends):
                    writer.write(l + '\n')
                    writer.write('{}\n'.format(len(X)))
                    for x_, y_ in zip(X, y):
                        writer.write('%.16f %.16f\n' % (x_, y_))
        else:
            with open(log, 'w') as writer:
                for X, y in data:
                    writer.write('\n')
                    writer.write('{}\n'.format(len(X)))
                    for x_, y_ in zip(X, y):
                        writer.write('%.16f %.16f\n' % (x_, y_))
    if 'DISPLAY' in os.environ:
        plt.show()


def get_data(path="audioset", padding='fix',
             train_noise=None, train_copy=1, split=None):
    logger = get_logger('utils')
    try:
        split = float(split)
    except:
        logger.info("split {} is not a float consider read from files"
                    .format(split))
    path = os.path.realpath(path)
    need_path = path + '/'

    recognized_padding_method = ['drop', 'fix']
    if padding in recognized_padding_method:
        need_path += '/{}-padding'.format(padding)
    else:
        logger.critical('Padding method can only be {}'
                        .format(recognized_padding_method))
        raise ValueError('unrecognized padding method {}'.format(padding))
    del recognized_padding_method

    if train_noise:
        need_path += '/noise{}-copy{}'.format(train_noise, train_copy)
    else:
        need_path += '/no-noise-copy{}'.format(train_copy)
    if isinstance(split, str):
        need_path += '/split-by-files'.format(split)
    elif isinstance(split, float):
        need_path += '/split{}'.format(split)
    else:
        need_path += '/no-split'

    if not os.path.exists(need_path):
        logger.info('Formatted audioset file not found, generating...')
        if not os.path.exists(path + '/raw'):
            logger.critical('Origin audioset files were not found in \"{}\"'
                            .format(path + '/raw'))
            raise IOError('Audioset files were not found')

        def process(data_path):
            data_dict = dict()
            logger.info('processing data path {}...'.format(data_path))
            for file_name in os.listdir(data_path):
                _, ext = os.path.splitext(os.path.join(data_path, file_name))
                if ext == '.gz':
                    with gzip.open(os.path.join(data_path, file_name),
                                   'rb') as readsub:
                        cur_file = pickle.load(readsub)
                        data_dict.update(cur_file)
                elif ext == '.npy':
                    cur_file = numpy.load(os.path.join(data_path, file_name))
                    data_dict.update(cur_file[()])
                else:
                    raise NotImplementedError('Unrecognized file extension {}'
                                              .format(ext))
            n_data = len(data_dict)
            keys = data_dict.keys()
            data_arr = [data_dict[key][0] for key in keys]

            labels = numpy.zeros((n_data, 527), dtype=numpy.int)
            for i, key in enumerate(keys):
                labels[i][data_dict[key][1]] = 1

            result_data, result_label, result_length = [], [], []
            for x, y in zip(data_arr, labels):
                result_length.append(x.shape[0])
                if padding == 'drop':
                    if x.shape[0] == 10:
                        result_data.append(x)
                        result_label.append(y)
                elif padding == 'fix':
                    x = numpy.concatenate(
                        (x, numpy.zeros((10 - x.shape[0], x.shape[1]))),
                        axis=0)
                    result_data.append(x)
                    result_label.append(y)
                else:
                    raise NotImplementedError('Unimplemented padding method {}'
                                              .format(padding))
            return result_data, result_label, result_length

        X_train, y_train, l_train = process(path + '/raw/train')
        X_test, y_test, l_test = process(path + '/raw/test')
        if not split:
            if train_noise:
                X_train = numpy.asarray(X_train)
                y_train = numpy.asarray(y_train)
                l_train = numpy.asarray(l_train)
                new_X_train = []
                new_y_train = []
                new_l_train = []
                for i in range(train_copy):
                    X_new = X_train.copy()
                    X_new += numpy.random.randn(*X_train.shape) * train_noise
                    new_X_train.append(X_new)
                    new_y_train.append(y_train.copy())
                    new_l_train.append(l_train.copy())
                X_train = numpy.concatenate(new_X_train, axis=0)
                y_train = numpy.concatenate(new_y_train, axis=0)
                l_train = numpy.concatenate(new_l_train, axis=0)
            train = DataSet(X_train, y_train, l_train)
            eval = DataSet([], [], [])
            test = DataSet(X_test, y_test, l_test)
            os.makedirs(need_path)
            with open(need_path + '/audioset.data', 'wb') as writer:
                joblib.dump((train, eval, test), writer)
            return train, eval, test
        else:
            if isinstance(split, float):
                X_train, X_eval, y_train, y_eval, l_train, l_eval = \
                    train_test_split(X_train, y_train, l_train, test_size=split)
            elif isinstance(split, str):
                X_eval, y_eval, l_eval = process(
                    os.path.realpath(split + '/raw/test'))
            else:
                raise NotImplementedError("Unrecognized split type of {}"
                                          .format(type(split)))
            if train_noise:
                X_train = numpy.asarray(X_train)
                y_train = numpy.asarray(y_train)
                l_train = numpy.asarray(l_train)
                new_X_train = []
                new_y_train = []
                new_l_train = []
                for i in range(train_copy):
                    X_new = X_train.copy()
                    X_new += numpy.random.randn(*X_train.shape) * train_noise
                    new_X_train.append(X_new)
                    new_y_train.append(y_train.copy())
                    new_l_train.append(l_train.copy())
                X_train = numpy.concatenate(new_X_train, axis=0)
                y_train = numpy.concatenate(new_y_train, axis=0)
                l_train = numpy.concatenate(new_l_train, axis=0)
            train = DataSet(X_train, y_train, l_train)
            eval = DataSet(X_eval, y_eval, l_eval)
            test = DataSet(X_test, y_test, l_test)
            os.makedirs(need_path)
            with open(need_path + '/audioset.data', 'wb') as writer:
                joblib.dump((train, eval, test), writer)
            return train, eval, test

    logger.info('Reading data from audioset...')
    with open(need_path + '/audioset.data', 'rb') as reader:
        train, eval, test = joblib.load(reader)
    logger.info("Data loaded")
    assert isinstance(train, DataSet)
    assert isinstance(test, DataSet)
    assert isinstance(eval, DataSet)
    return train, eval, test


def metrics(answer, output):
    ap = average_precision_score(answer, output)
    auc = roc_auc_score(answer, output)
    return auc, ap


def set_seed(seed=None):
    if seed is None:
        seed = numpy.random.randint(0, 65535)
    numpy.random.seed(seed)
    tensorflow.set_random_seed(seed)
    mxnet.random.seed(seed)
    return seed
