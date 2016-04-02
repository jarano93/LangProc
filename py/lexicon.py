#!/usr/bin/python

import re
import numpy as np

class Vocabulary:
    # TODO change it so the re split works

    reg_str = '!?;,.-_+=^&%:[]()\{\}\"@'

    def __init__(self, text, save=True):
        words = list(set(re.split(self.reg_str, text.lower())))
        self.len_vocab = len(words)
        self.word_key = dict(zip(words, range(self.len_vocab)))
        self.int_key = dict(zip(range(self.len_vocab), words)

    def word(self, val):
        return self.int_keys[num]

    def num(self, word):
        return self.vocab_key[word]

    def hcol(self, word):
        hcol = np.zeros((self.len_vocab, 1))
        hcol[self.word_key[word]] = 1
        return hcol
