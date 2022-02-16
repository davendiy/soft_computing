#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 2021-12-01
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from itertools import product
from typing import Callable, List

import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def tanh_deriv(tanh_x: np.ndarray) -> np.ndarray:
    return 1 - tanh_x ** 2


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(x))


def sigmoid_deriv(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)


def mse(y: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return ((y - y_true) ** 2).mean()


def mse_deriv(y: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return 2 * (y - y_true)

def fn(x: np.ndarray) -> np.ndarray:
    return np.sin(x[:, 0]) + np.sin(x[:, 1]) - np.cos(x[:, 2])


def normalize_sigmoid(x: np.ndarray) -> np.ndarray:
    x_max = x.max()
    x_min = x.min()
    return (x - x_min) / (x_max - x_min)


x_1 = np.array([7, 8, 9])
x_2 = np.array([4, 5, 6])
x_3 = np.array([2, 3, 4])

# prepare x
x = np.vstack(list(product(x_1, x_2, x_3))).astype('f')
# calculate y
y_1 = fn(x)
# normalize y
y_max, y_min = y_1.max(), y_1.min()
y_1_normalized = (y_1 - y_min) / (y_max - y_min)

# calculate
y_mean = np.mean(y_1_normalized)
y_2 = np.zeros_like(y_1_normalized)
y_2[y_1_normalized > y_mean] = 1
y = np.vstack([y_1_normalized,  y_2]).T

for i in range(x.shape[1]):
      x[:, i] = normalize_sigmoid(x[:, i])

np.hstack([x, y])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=20)


def forward(x, parameters, history, y=None, validation=False):
    hidden_weights = parameters['hidden_weights']
    output_weights = parameters['output_weights']
    input_layer_outputs = np.hstack((np.ones((x.shape[0], 1)), x))
    hidden_layer_outputs = np.hstack(
        (np.ones((x.shape[0], 1)), sigmoid(np.dot(input_layer_outputs, hidden_weights))))
    output_layer_outputs = np.dot(hidden_layer_outputs, output_weights)
    if validation:
        loss = mse(output_layer_outputs, y)
        history['val loss'].append(loss)
    else:
        parameters['input_layer_outputs'] = input_layer_outputs
        parameters['hidden_layer_outputs'] = hidden_layer_outputs
        parameters['output_layer_outputs'] = output_layer_outputs


def backward(y, parameters, history):
    output_layer_outputs = parameters['output_layer_outputs']
    hidden_weights = parameters['hidden_weights']
    output_weights = parameters['output_weights']
    hidden_layer_outputs = parameters['hidden_layer_outputs']
    input_layer_outputs = parameters['input_layer_outputs']
    loss = mse(output_layer_outputs, y)
    history['training loss'].append(loss)
    output_error = mse_deriv(output_layer_outputs, y)

    hidden_error = sigmoid_deriv(hidden_layer_outputs[:, 1:]) * np.dot(output_error, output_weights.T[:, 1:])

    # partial derivatives
    hidden_pd = input_layer_outputs[:, :, np.newaxis] * hidden_error[:, np.newaxis, :]
    output_pd = hidden_layer_outputs[:, :, np.newaxis] * output_error[:, np.newaxis, :]

    total_hidden_gradient = np.average(hidden_pd, axis=0)
    total_output_gradient = np.average(output_pd, axis=0)

    # update weights
    hidden_delta = - alpha * (total_hidden_gradient * (1 - mu) + mu * prew_hidden_delta)
    output_delta = - alpha * (total_output_gradient * (1 - mu) + mu * prew_output_delta)

    hidden_weights += hidden_delta
    output_weights += output_delta
    parameters.update({
        'hidden_weights': hidden_weights,
        'output_weights': output_weights
    })

num_inputs = 3
num_hidden = 3
num_outputs = 2
alpha = 0.001
mu = 0.01

history = {"training loss": [], "val loss": []}

hidden_weights = np.random.normal(0, 1, size=(num_inputs + 1, num_hidden))
output_weights = np.random.normal(0, 1, size=(num_hidden + 1, num_outputs))

prew_hidden_delta = 0
prew_output_delta = 0

num_iterations = 1000
parameters = {'hidden_weights': hidden_weights,
              'output_weights': output_weights}

for i in tqdm(range(num_iterations)):
    # forward
    forward(x_train, parameters, history)
    # backward
    backward(y_train, parameters, history)
    # validation
    forward(x_test, parameters, history, validation=True, y=y_test)

plt.plot(range(num_iterations), history['training loss'], label='Training loss')
plt.plot(range(num_iterations), history['val loss'], label='Validation loss')
plt.legend()
