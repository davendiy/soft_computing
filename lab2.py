#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 2021-12-01
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from softutils.nn import NeuralNetwork, SymbolicMatrix
import softutils as sf
import numpy as np
# import matplotlib.pyplot as plt


class Model(NeuralNetwork):

    def __init__(self, N, hidden, out):
        self._w1 = SymbolicMatrix.create_W(hidden, N, 'w1')
        self._b1 = SymbolicMatrix.create_b(hidden, 'b1')
        self._w2 = SymbolicMatrix.create_W(out, hidden, 'w2')
        self._b2 = SymbolicMatrix.create_b(out, 'b2')
        self._activation1 = sf.relu
        self._activation2 = sf.tanh
        self._all_params = self._w1.flatten() \
                                + self._w2.flatten() \
                                + self._b1.flatten() \
                                + self._b2.flatten()

    def __call__(self, input_tensor: SymbolicMatrix) -> SymbolicMatrix:
        res1 = self._w1 @ input_tensor.T() + self._b1
        res1 = res1.apply(self._activation1)
        res2 = self._w2 @ res1+ self._b2
        res2 = res2.apply(self._activation2)
        return res2[0]

    def all_parameters(self):
        return [str(el) for el in self._all_params]


def target(x1, x2, x3):
    return np.sin(x1) + np.sin(x2) - np.cos(x3)


test_model = Model(3, 2, 1)
train_x = [
    [7, 4, 2],
    [8, 5, 3],
    [9, 6, 4],
]

train_y = [[target(*x)] for x in train_x]

train_x = np.array(train_x, dtype='double')
train_y = np.array(train_y, dtype='double')
train_x = (train_x - train_x.mean()) / train_x.std()
train_y = (train_y - train_y.mean()) / train_y.std()

history = test_model.fit(train_x, train_y, eta=5000, max_iterations=20)
print(history)
