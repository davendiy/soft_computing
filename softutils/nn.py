#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 2021-11-29
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from __future__ import annotations

from ._function import *
from ._function import _Var
import abc


class SymbolicMatrix:

    def __init__(self, matrix):
        self._body = matrix
        if not isinstance(matrix[0], list):
            self._body = [matrix]

    def __matmul__(self, other):
        return self.dot(other)

    @property
    def shape(self):
        return [len(self._body), len(self._body[0])]

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._body[item]
        elif isinstance(item, tuple):
            return self._body[item[0]][item[1]]
        else:
            raise NotImplementedError()

    def __iter__(self):
        return iter(self._body)

    def dot(self, other: SymbolicMatrix) -> SymbolicMatrix:
        res_matrix = []
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Can't multiply matrices with shapes {self.shape} and {other.shape}")
        for i in range(self.shape[0]):
            res_matrix.append([])
            for j in range(other.shape[1]):
                res_matrix[i].append(self._dot_vectors(self[i], [row[j] for row in other]))
        return SymbolicMatrix(res_matrix)

    @staticmethod
    def _dot_vectors(vec1, vec2):
        return sum(x1 * x2 for x1, x2 in zip(vec1, vec2))

    def T(self) -> SymbolicMatrix:
        return SymbolicMatrix([
            [self[i, j] for i in range(self.shape[0])]
            for j in range(self.shape[1])
        ])

    def __add__(self, other: SymbolicMatrix):
        return SymbolicMatrix([
            [x + y for x, y in zip(row1, row2)]
            for row1, row2 in zip(self, other)
        ])

    def __str__(self):
        return str(self._body)

    def __repr__(self):
        return f'SymbolicMatrix({str(self)})'

    @classmethod
    def create_W(cls, n_in, n_out, name):
        return cls([
            [Var(f'{name}_{i}_{j}') for j in range(n_out)]
            for i in range(n_in)
        ])

    @classmethod
    def create_b(cls, n, name):
        return cls([[Var(f'{name}_{i}')] for i in range(n)])

    def apply(self, f):
        return SymbolicMatrix([
            [f(x) for x in row]
            for row in self
        ])

    def flatten(self):
        return [self[i][j] for j in range(self.shape[1]) for i in range(self.shape[0])]


class NeuralNetwork(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, input_tensor: SymbolicMatrix):
        pass

    @abc.abstractmethod
    def all_parameters(self) -> typ.List[_Var]:
        pass

    def fit(self, train_x, train_y, loss='mse', eta=0.1, algo='gradient_descent',
            max_iterations=100, eps=10e-5):
        if loss != 'mse':
            raise NotImplementedError()

        res = Const(0)
        for input, output in zip(train_x, train_y):
            nn_output = self(SymbolicMatrix(input))
            for y1, y2 in zip(output, nn_output):
                res += (y1 - y2) ** 2
        history, params = res.minimize(*self.all_parameters(), eta=eta, algo=algo,
                     max_iterations=max_iterations, eps=eps)
        for var, value in zip(self.all_parameters(), params):
            Var(var).value = value
        return history

    def predict(self, input_tensor):
        return [res() for res in self(input_tensor)]
