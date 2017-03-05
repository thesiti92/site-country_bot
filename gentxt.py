#!/usr/bin/env python
"""Example to generate text from a recurrent neural network language model.

This code is ported from following implementation.
https://github.com/longjie/chainer-char-rnn/blob/master/sample.py

"""
import argparse
import sys

import numpy as np
import six
import json
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import serializers
import requests
import BytesIO

class RNNForLM(chainer.Chain):

    def __init__(self, n_vocab, n_units, train=True):
        super(RNNForLM, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=self.train))
        h2 = self.l2(F.dropout(h1, train=self.train))
        y = self.l3(F.dropout(h2, train=self.train))
        return y


def gentxt( primetext="when", seed = 123, unit=650, sample = 1, length = 30):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', '-m', type=str, required=True,
    #                     help='model data, saved by train_ptb.py')
    # parser.add_argument('--primetext', '-p', type=str, required=True,
    #                     default='',
    #                     help='base text data, used for text generation')
    # parser.add_argument('--seed', '-s', type=int, default=123,
    #                     help='random seeds for text generation')
    # parser.add_argument('--unit', '-u', type=int, default=650,
    #                     help='number of units')
    # parser.add_argument('--sample', type=int, default=1,
    #                     help='negative value indicates NOT use random choice')
    # parser.add_argument('--length', type=int, default=20,
    #                     help='length of the generated text')
    # parser.add_argument('--gpu', type=int, default=-1,
    #                     help='GPU ID (negative value indicates CPU)')
    gpu=-1
    # args = parser.parse_args()

    np.random.seed(seed)

    xp = cuda.cupy if gpu >= 0 else np

    # load vocabulary
    vocab = json.load(open("data/vocab_indexes.json"))
    ivocab = {}
    for c, i in vocab.items():
        ivocab[i] = c

    # should be same as n_units , described in train_ptb.py
    n_units = unit

    lm = RNNForLM(len(vocab), n_units, train=False)
    model = L.Classifier(lm)
    request = requests.get(https://s3.us-east-2.amazonaws.com/country-bot/650u_20p_100e.dat)
    serializers.load_npz(BytesIO(request.content), model)
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    model.predictor.reset_state()

    primetext = primetext
    if isinstance(primetext, six.binary_type):
        primetext = primetext.decode('utf-8')

    if primetext in vocab:
        prev_word = chainer.Variable(xp.array([vocab[primetext]], xp.int32))
    else:
        return "unknown"
        exit()
    text = []
    prob = F.softmax(model.predictor(prev_word))
    text.append(primetext + ' ')

    for i in six.moves.range(length):
        prob = F.softmax(model.predictor(prev_word))
        if sample > 0:
            probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
            probability /= np.sum(probability)
            index = np.random.choice(range(len(probability)), p=probability)
        else:
            index = np.argmax(cuda.to_cpu(prob.data))

        if ivocab[index] == '<eos>':
            text.append('. ')
        else:
            text.append(ivocab[index] + ' ')

        prev_word = chainer.Variable(xp.array([index], dtype=xp.int32))
    return "".join(text)

if __name__ == '__main__':
    print gentxt()
