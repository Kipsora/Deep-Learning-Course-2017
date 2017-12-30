from __future__ import print_function

import argparse

import progressbar

from project import utils
from project.datasets import Audio
from project.hparam import get_hparam
from project.models.tensorflow import DeepNN


def main(config):
    dataset = Audio('dataset/washed/audioset.data')

    hparam = DeepNN.default_hparam()
    hparam.layers = [
        get_hparam(
            units=40,
            activation='leaky_relu',
            batchnorm=get_hparam(),
            dropout=get_hparam(
                rate=0.5
            )
        ),
        get_hparam(
            units=100,
            activation='leaky_relu',
            batchnorm=get_hparam(),
            dropout=get_hparam(
                rate=0.5
            )
        ),
        get_hparam(
            units=100,
            activation='leaky_relu',
            batchnorm=get_hparam(),
            dropout=get_hparam(
                rate=0.5
            )
        ),
        get_hparam(
            units=100,
            activation='leaky_relu',
            batchnorm=get_hparam(),
            dropout=get_hparam(
                rate=0.5
            )
        ),
        get_hparam(
            units=40,
            activation='leaky_relu',
            batchnorm=get_hparam(),
            dropout=get_hparam(
                rate=0.5
            )
        )
    ]
    hparam.layers.append(get_hparam(
        units=dataset.osize,
        batchnorm=get_hparam()
    ))
    hparam.loss = 'mean_squared_error'
    hparam.optimizer.type = 'Adam'
    hparam.optimizer.params.learning_rate = 0.0001
    for k, v in hparam.iterall('hparam.'):
        print(k, '=', v)
    model = DeepNN('test', dataset.ishape, dataset.osize, hparam)

    if config.load:
        model.load(config.load)
    else:
        config.load = 1
    pbar = progressbar.ProgressBar(widgets=[
        'Progress: ', progressbar.Percentage(), ' ', progressbar.Timer(), ' ',
        progressbar.ETA(), ' ', ''
    ], maxval=config.epoch).start()

    X_eval, y_eval = dataset.eval
    X_eval = X_eval[:5000]
    y_eval = y_eval[:5000]

    for epoch in range(config.load, config.load + config.epoch):
        model.train(*dataset.batch(config.batch_size))
        if epoch == config.load or epoch % config.print_period == 0 or \
                                epoch + 1 == config.load + config.epoch:
            predicts = model.predict(X_eval)
            auc, ap = utils.metrics(y_eval, predicts)
            pbar.widgets[-1] = 'Train Loss: %.8f Eval Metrics: (Loss: %.8f, AUC' \
                               ': %.8f, AP: %.8f)' % (
                model.loss(X_eval, y_eval),
                model.loss(*dataset.batch(config.batch_size)),
                auc, ap
            )
            pbar.update(epoch - config.load)
    pbar.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--print-period', type=int, default=300)
    main(parser.parse_args())
