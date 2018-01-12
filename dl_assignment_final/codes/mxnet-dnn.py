from __future__ import print_function

import argparse

import mxnet
import mxnet.autograd

from project.utils import get_data, metrics, set_seed


class DNN(mxnet.gluon.nn.Block):
    def __init__(self):
        super(DNN, self).__init__()

        self.batch1 = mxnet.gluon.nn.BatchNorm()
        self.dense1 = mxnet.gluon.nn.Dense(600)
        self.dropo1 = mxnet.gluon.nn.Dropout(0.5)

        self.dense2 = mxnet.gluon.nn.Dense(527)

    def forward(self, inputs):
        hidden = inputs.reshape([-1, 1280])
        hidden = self.batch1(hidden)
        hidden = self.dense1(hidden)
        hidden = mxnet.nd.relu(hidden)
        hidden = self.dropo1(hidden)

        hidden = self.dense2(hidden)

        return hidden

class Loss(mxnet.gluon.loss.Loss):
    def hybrid_forward(self, F, pred, label):
        sig_pred = F.sigmoid(pred)
        return -F.mean(F.sum(label * F.log(sig_pred) +
                             (1 - label) * F.log(1 - sig_pred), axis=1))

    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(Loss, self).__init__(weight, batch_axis, **kwargs)


def main(config):
    print('seed', '=', set_seed(config.seed))

    net = DNN()
    criterion = Loss()
    net.initialize(init=mxnet.initializer.Normal(sigma=config.init_stddev),
                   ctx=mxnet.gpu())
    trainer = mxnet.gluon.Trainer(net.collect_params(), 'rmsprop', {
        'learning_rate': config.learning_rate,
        'wd': config.reg
    })
    if config.load_from:
        load_from = config.load_from
        # TODO: Load from models
    else:
        load_from = 0

    train_set, eval_set, test_set = get_data(
        path='audioset/small',
        split=config.split,
        train_noise=config.train_noise,
        train_copy=config.train_copy)
    print(train_set.shape)
    print(eval_set.shape)
    print(test_set.shape)

    eval_X = mxnet.nd.array(eval_set.X, ctx=mxnet.gpu())
    eval_y = mxnet.nd.array(eval_set.y, ctx=mxnet.gpu())
    test_X = mxnet.nd.array(test_set.X, ctx=mxnet.gpu())
    test_y = mxnet.nd.array(test_set.y, ctx=mxnet.gpu())

    for epoch in range(load_from + 1, load_from + config.n_epoch + 1):
        X_batch, y_batch, l_batch = train_set.batch(size=config.batch_size)

        X_batch = mxnet.nd.array(X_batch, mxnet.gpu())
        y_batch = mxnet.nd.array(y_batch, mxnet.gpu())

        with mxnet.autograd.record():
            results = net(X_batch.as_in_context(mxnet.gpu()))
            train_loss = criterion(results, y_batch)
        train_loss.backward()
        trainer.step(batch_size=config.batch_size)

        if epoch % config.print_period == 0 or epoch == load_from:
            results = net(eval_X.as_in_context(mxnet.gpu()))
            eval_predicts = mxnet.nd.sigmoid(results)
            eval_loss = criterion(results, eval_y.as_in_context(mxnet.gpu()))
            auc, ap = metrics(eval_set.y, eval_predicts.asnumpy())
            print('epoch {}/{} train loss: {} eval: loss: {} auc: {} ap: {}'
                  .format(epoch, load_from + config.n_epoch,
                          train_loss.asscalar(), eval_loss.asscalar(),
                          auc, ap))

    results = net(test_X.as_in_context(mxnet.gpu()))
    test_predicts = mxnet.nd.sigmoid(results)
    test_loss = criterion(results, test_y.as_in_context(mxnet.gpu()))
    auc, ap = metrics(test_set.y, test_predicts.asnumpy())

    print('test: loss: {} auc: {} ap: {}'
          .format(test_loss, auc, ap))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--print_period', default=100, type=int)
    parser.add_argument('--n_epoch', default=2000, type=int)
    parser.add_argument('--load_from', default=None, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--clip_by_norm', default=5.0, type=float)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--init_stddev', default=0.001, type=float)
    parser.add_argument('--reg', default=0.00, type=float)
    parser.add_argument('--train_noise', default=0.0, type=float)
    parser.add_argument('--train_copy', default=1, type=int)
    parser.add_argument('--split', default='audioset/small/raw/test', type=str)
    parser.add_argument('--seed', default=None, type=int)
    main(parser.parse_args())
