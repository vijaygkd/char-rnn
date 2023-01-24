'''
Utils for data prep
'''
import numpy as np
import torch


def str_to_ids(text):
    '''Convert string to ASCII code vectors'''
    return [ord(c) for c in text]


def char_idx_to_str(char_idx):
    '''Convert vector to char indices to string'''
    return ''.join([chr(c) for c in char_idx])


def process_text(x):
    x = '$' + x
    y = x[1:] + '$'
    
    x_vec = str_to_ids(x)
    y_vec = str_to_ids(y)
    return x_vec, y_vec


def process_corpus(corpus, seq_len):
    """corpus is list of docs. Return x and y tensors"""
    pad_idx = ord(' ')
    data_shape = (len(corpus), seq_len)
    x_arr = np.full(data_shape, pad_idx)
    y_arr = np.full(data_shape, pad_idx)
    
    for i, doc in enumerate(corpus):
        doc = doc[:seq_len]
        x_vec, y_vec = process_text(doc)
        x_arr[i][0:len(x_vec)] = x_vec
        y_arr[i][0:len(y_vec)] = y_vec
    
    X = torch.tensor(x_arr)
    Y = torch.tensor(y_arr)
    
    return X, Y
