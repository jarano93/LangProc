#!/usr/bin/python

import numpy as np
import numpy.linalg as la
import numpy.random as nr
import nltk
from collections import Counter
import py.onehot as oh

# TODO import nltk

'''
    This code has functions that preprocesses a corpus for embedding words in a
        dense vector space,
    finding the word embedding matrix for the words in the corpus,
    and converting a dataset of words into onehots and then dense vectors
    Uses Mikolov's Skip Gram method
'''

def embed(str_dataset, vect_len):
    pass

def prep(str_dataset, s_threshold, c_rad):
    tokens = nltk.word_tokenize(str_dataset)
    sub_tokens = prep_subsample(tokens, s_threshold)
    int_seq, hot_len = prep_intify(sub_tokens)
    wc_data = prep_context(int_seq, c_rad)
    return wc_data, hot_len

def prep_intify(t_list):
    uniques = set(t_list)
    len_uniqes = len(uniques)
    key = dict(zip(uniques, range(len_uniques))
    seq = np.zeros(len_tokens)
    for i in xrange(len(t_list)):
        seq[i] = key[tokens[i]]
    return seq, len_uniques

def prep_subsample(t_list, s_threshold):
    t_len = len(t_list)
    count = Counter(t_list)
    s_list = []
    for i in xrange(t_len):
        if nr.sample() > 1 - np.sqrt(s_threshold / count[t_list[i]]):
            s_list.append(t_list[i])
    return s_list

def prep_context(seq, c_rad):
    wc_data = []
    s_len = len(seq)
    for i in xrange(s_len):
        word = seq[i]
        c_min =  i - c_rad
        c_min = 0 if c_min < 0 else c_min
        c_max = i + c_rad
        c_max = s_len if c_max > s_len else c_max
        c_list = range(c_min, c_max)
        c_list.remove(i)
        context = []
        for j in xrange(len(c_list)):
            context.append(seq[c_list[j]])
        wc_data.append((word, context))
    return wc_data

def optimize_matrix(matrix, wc_data, hot_len, N, TOL):
    mem = np.zeros_like(matrix)
    old = matrix
    for n in xrange(N):
        grad = np.zeros_like(matrix)

        # actual math here
        for i in xrange(len(wc_data):
            

        # adagrad
        mem += np.square(grad)
        old = new
        new = new - grad / np.sqrt(1e-8 + mem)
        if la.norm(np.absolute(new - old), 2) < TOL:
            break
    return new
