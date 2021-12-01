#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 2021-11-30
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

import abc
import numpy as np

import plotly.graph_objs as go


class Function(metaclass=abc.ABCMeta):
    global_min: np.ndarray = ...
    bounds: np.ndarray = ...
    name: str = ...
    dim: int = ...

    # singleton pattern
    __defined = None

    def __new__(cls, *args, **kwargs):
        if cls.__defined is None:
            cls.__defined = super(Function, cls).__new__(cls, *args, **kwargs)
        return cls.__defined

    @abc.abstractmethod
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        pass

    def plot(self, points_amount=100, min_trace=None):
        (x1_left, x1_right), (x2_left, x2_right) = self.bounds
        x1 = np.linspace(x1_left, x1_right, points_amount)
        x2 = np.linspace(x2_left, x2_right, points_amount)
        x1v, x2v = np.meshgrid(x1, x2)
        f = self(x1v, x2v)

        data = [go.Surface(z=f, x=x1v, y=x2v, surfacecolor=-f)]
        if min_trace is not None:
            x1 = min_trace[:, 0].flatten()
            x2 = min_trace[:, 1].flatten()
            f = self(x1, x2).flatten()
            data.append(go.Scatter3d(x=x1,
                                     y=x2,
                                     z=f,
                                     mode='lines+markers', name='Minimization trace',
                                     marker=dict(size=2, color=-f, colorscale='Blues')))
        fig = go.Figure(data=data)
        fig.update_layout(title=self.name)
        fig.show()


class Sphere(Function):
    global_min = np.array([0., 0., 0.])
    bounds = np.array([[-10, 10], [-10, 10]])
    name = 'Sphere function'
    dim = 2

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return x1 ** 2 + x2 ** 2


class Rozenbrock(Function):
    name = 'Rozenbrock function'
    global_min = np.array([1., 1., 0.])
    bounds = np.array([[-10, 10], [-10, 10]])
    dim = 2

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2


class Isome(Function):
    name = 'Isome function'
    global_min = np.array([np.pi, np.pi, -1.])
    bounds = np.array([[-10, 10], [-10, 10]])
    dim = 2

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi) ** 2 + (x2 - np.pi) ** 2))
