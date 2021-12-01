#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 2021-11-30
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

import abc
import numpy as np

import plotly.graph_objs as go


class MinimizationAlgorithm(metaclass=abc.ABCMeta):

    def __init__(self, target):
        self.target = target

    @abc.abstractmethod
    def fit(self, num_iterations) -> (np.ndarray, np.ndarray):
        pass

    def show_results(self, history, min_trace, points_amount=100):
        num_iterations = len(history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(num_iterations)),
                                 y=history[:, 0], mode='lines', name='average_pop_loss'))
        fig.add_trace(go.Scatter(x=list(range(num_iterations)),
                                 y=history[:, 1], mode='lines', name='best_sample_loss'))
        fig.show()

        self.target.plot(min_trace=min_trace, points_amount=points_amount)
