import os
import pickle
import tensorflow
import numpy
import mxnet
from sklearn.metrics import roc_auc_score, average_precision_score


def set_seed(seed=None):
    tensorflow.set_random_seed(seed)
    numpy.random.seed(seed)
    if seed:
        mxnet.random.seed(seed)


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


def save(o, path):
    with open(path, 'wb') as writer:
        pickle.dump(o, writer)


def load(path):
    with open(path, 'rb') as reader:
        return pickle.load(reader)


def metrics(answer, output):
    ap = average_precision_score(answer, output)
    auc = roc_auc_score(answer, output)
    return auc, ap
