# This script is a toy script which gives you basic idea of loading the data provided
# Read all the bal_train data into dicts

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import pickle
import gzip
from os import listdir, path


# load data to a whole dict
# dict{key:[data, label]}

def read_dir(filedir='./eval'):
    data_dict = {}
    for subfile in listdir(filedir):
        with gzip.open(path.join(filedir,subfile), 'rb') as readsub:
            curfile = pickle.load(readsub)
            data_dict = dict(data_dict.items()+curfile.items())
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

def eval(pred, label):
    print(pred, label)
    mAP = average_precision_score(label, pred)
    mAUC = roc_auc_score(label, pred)
    print 'mAP:',mAP
    print 'mAUC:',mAUC

if __name__ == "__main__":
    train_data_dict = read_dir('./train')
    eval_data_dict = read_dir('./eval')    

    train_data, train_label = format_data(train_data_dict)
    eval_data, eval_label = format_data(eval_data_dict)

    #shape of the prediction array
    for d in eval_data:
        print d.shape
    print("eval_data.shape:", eval_data.shape)
    print("train_data.shape:", train_data.shape)
    print("eval_label.shape:", eval_label.shape)
    print("train_label.shape:", train_label.shape)
    pred = np.random.rand(len(eval_data_dict), 527)

    #save the output
    with open('output.pkl','w') as f: 
        pickle.dump(pred,f)   

    eval(pred, eval_label)