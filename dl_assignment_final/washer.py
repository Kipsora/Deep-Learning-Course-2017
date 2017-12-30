import gzip
import numpy as np
import pickle
import os


def read_dir(filedir='./eval'):
    data_dict = {}
    for subfile in os.listdir(filedir):
        with gzip.open(os.path.join(filedir, subfile), 'rb') as readsub:
            curfile = pickle.load(readsub)
            data_dict = dict(data_dict.items() + curfile.items())
    return data_dict


def format_data(data_dict):
    n_data = len(data_dict)
    keys = data_dict.keys()
    data_arr = [data_dict[key][0] for key in keys]
    labels = [data_dict[key][1] for key in keys]
    data_arr = np.array(data_arr)
    label_arr = np.zeros((n_data, 527), dtype=np.int)
    for i in range(len(labels)):
        for idx in labels[i]:
            label_arr[i][idx] = 1
    return data_arr, label_arr


if __name__ == '__main__':
    if not os.path.exists('dataset/washed/audioset.data'):
        if not os.path.exists('dataset/washed'):
            os.makedirs('dataset/washed')

        train_data_dict = read_dir('dataset/raw/audioset/train')
        eval_data_dict = read_dir('dataset/raw/audioset/eval')
        train_data, train_label = format_data(train_data_dict)
        eval_data, eval_label = format_data(eval_data_dict)
        del train_data_dict
        del eval_data_dict

        X_train = []
        y_train = []
        X_eval = []
        y_eval = []
        for x, y in zip(train_data, train_label):
            if x.shape[0] == 10:
                X_train.append(x)
                y_train.append(y)
        for x, y in zip(eval_data, eval_label):
            if x.shape[0] == 10:
                X_eval.append(x)
                y_eval.append(y)
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_eval = np.asarray(X_eval)
        y_eval = np.asarray(y_eval)

        print("X_train.shape:", X_train.shape)
        print("y_train.shape:", y_train.shape)
        print("X_eval.shape:", X_eval.shape)
        print("y_eval.shape:", y_eval.shape)

        with open('dataset/washed/audioset.data', 'wb') as writer:
            pickle.dump(
                (X_train, y_train, X_eval, y_eval),
                writer
            )
