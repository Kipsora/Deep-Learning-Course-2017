import glob

import numpy
from sklearn.externals import joblib

from project.utils import get_data, metrics

if __name__ == '__main__':
    train_set, eval_set, test_set = get_data(
        path='audioset/small',
        split=0.3,
        train_noise=0,
        train_copy=1)
    predicts = []
    for path in glob.glob('*.predicts'):
        print('Loading from', path)
        with open(path, 'rb') as reader:
            predicts.append(joblib.load(reader))
    predicts = numpy.asarray(predicts)
    predicts = numpy.mean(predicts, axis=0)
    auc, ap = metrics(test_set.y, predicts)
    with open("final_output.pkl", 'wb') as writer:
        numpy.save(writer, predicts)
    print("AUC={}, AP={}".format(auc, ap))
