#!/usr/bin/python

import numpy as np
import numpy.linalg as la
import numpy.random as nr
import random as r
import nltk # on the cutting block.  Maybe just split(' ') to tokenize
from collections import Counter
import py.onehot as oh

'''
    This code has functions that preprocesses a corpus for embedding words in a
        dense vector space,
    finding the word embedding matrix for the words in the corpus,
    and converting a dataset of words into onehots and then dense vectors
    Uses Mikolov's Skip Gram method
'''

def sig(x):
    return 1 / (1 + np.exp(-x))

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
        if r.random() > 1 - np.sqrt(s_threshold / count[t_list[i]]):
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

def negative_sample(num_samples, c_rad, max_val):
    # generates negative samples based on the dataset
    # random samples tbh fam
    neg_data = []
    for i in xrange(num_samples):
        key = r.randrange(max_val)
        context = []
        for j in xrange(c_rad * 2):
            context.append(r.randrange(max_val))
        neg_data.append((key, context))
    return neg_data

def optimize_matrix(matrix, wc_data, neg_data, hot_len, N, TOL):
    mem = np.zeros_like(matrix)
    old = matrix
    for n in xrange(N):
        grad = np.zeros_like(matrix)

        # train for real samples from the data
        for i in xrange(len(wc_data):
            word = oh.hcol(wc_data[i][0], hot_len)
            em_word = np.dot(matrix, word)
            contexts = wc_data[i][1]
            for j in xrange(len(contexts):
                cont = oh.hcol(contexts[j], hot_len)
                em_cont = np.dot(matrix, cont)
                sigmoid = sig(np.dot(em_word.T, em_cont))
                delta = (sigmoid - 1) / sigmoid
                delta *= np.dot(em_word, cont.T) + np.dot(em_cont, word.T)
                grad += delta

        # train against negative samples
        for i in xrange(len(neg_data):
            nord = oh.hcol(wc_data[i][0], hot_len_
            em_nord = np.dot(matrix, nord)
            nontexts = wc_data[i][1]
            for j in xrange(len(nontexts):
                nont = oh.hcol(nontexts[j], hot_len)
                em_nont = np.dot(matrix, nont)
                sigmoid = sig(np.dot(-em_nord.T, em_nont))
                delta = (1 - sigmoid) / sigmoid
                delta *= np.dot(em_nord, nont.T) + np.dot(em_nont, nord.T)
                grad += delta

        # adagrad
        mem += np.square(grad)
        old = new
        new = new - grad / np.sqrt(1e-8 + mem)
        if la.norm(np.absolute(new - old), 2) < TOL:
            return new
    print "Unconverged after %d iterations" % (N + 1)
    return new
