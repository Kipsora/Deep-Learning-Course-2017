from __future__ import print_function

import argparse

import torch.nn
import torch.optim
import torch.nn.functional
import torch.autograd

from project.utils import get_data, metrics, set_seed


class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.batch1 = torch.nn.BatchNorm1d(1280)
        self.dense1 = torch.nn.Linear(1280, 600)
        self.dropo1 = torch.nn.Dropout()

        self.dense2 = torch.nn.Linear(600, 527)

    def forward(self, inputs):
        hidden = inputs.view(-1, 1280)

        hidden = self.batch1(hidden)
        hidden = self.dense1(hidden)
        hidden = torch.nn.functional.relu(hidden)
        hidden = self.dropo1(hidden)

        hidden = self.dense2(hidden)
        hidden = torch.nn.functional.sigmoid(hidden)

        return hidden


def main(config):
    print('seed', '=', set_seed(config.seed))

    net = DNN()
    net.cuda()
    optimizer = torch.optim.RMSprop(net.parameters(), config.learning_rate)

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

    criterion = torch.nn.BCELoss()
    eval_X = torch.autograd.Variable(torch.from_numpy(eval_set.X).cuda())
    eval_y = torch.autograd.Variable(torch.from_numpy(eval_set.y).cuda())
    test_X = torch.autograd.Variable(torch.from_numpy(test_set.X).cuda())
    test_y = torch.autograd.Variable(torch.from_numpy(test_set.y).cuda())

    for epoch in range(load_from + 1, load_from + config.n_epoch + 1):
        X_batch, y_batch, l_batch = train_set.batch(size=config.batch_size)

        X_batch = torch.from_numpy(X_batch).cuda()
        y_batch = torch.from_numpy(y_batch).cuda()

        X_batch = torch.autograd.Variable(X_batch)
        y_batch = torch.autograd.Variable(y_batch)

        optimizer.zero_grad()
        outputs = net(X_batch)
        train_loss = criterion(outputs, y_batch.float())
        train_loss.backward()
        optimizer.step()

        if epoch % config.print_period == 0 or epoch == load_from:
            eval_predicts = net(eval_X)
            eval_loss = criterion(eval_predicts, eval_y.float())
            auc, ap = metrics(eval_set.y, eval_predicts.data)
            print('epoch {}/{} train loss: {} eval: loss: {} auc: {} ap: {}'
                  .format(epoch, load_from + config.n_epoch,
                          train_loss, eval_loss, auc, ap))

    test_predicts = net(test_X)
    test_loss = criterion(test_predicts, test_y.float())
    auc, ap = metrics(test_set.y, test_predicts.data)

    print('test: loss: {} auc: {} ap: {}'
          .format(test_loss, auc, ap))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--print_period', default=100, type=int)
    parser.add_argument('--n_epoch', default=2000, type=int)
    parser.add_argument('--load_from', default=None, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--clip_by_norm', default=5.0, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--init_stddev', default=0.01, type=float)
    parser.add_argument('--reg', default=0.01, type=float)
    parser.add_argument('--train_noise', default=0.0, type=float)
    parser.add_argument('--train_copy', default=1, type=int)
    parser.add_argument('--split', default='audioset/small/raw/test', type=str)
    parser.add_argument('--seed', default=None, type=int)
    main(parser.parse_args())
