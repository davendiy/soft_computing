#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 2021-12-01
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (30, 10)
i = 4


def point_slope_fn_factory(x_1, y_1, alpha, slope_type):
    assert slope_type in ["left", "right"]
    x_0 = x_1 + (- alpha if slope_type == "left" else alpha)
    if x_0 != x_1:
        slope = - y_1 / (x_0 - x_1)

        def _fn(x):
            return y_1 + slope * (x - x_1)
    else:
        def _fn(x):
            return y_1 if x == x_0 else 0
    return _fn, x_0


class TrapezoidFunciton:
    def __init__(self, m_low, m_high, alpha, beta, h, name):
        self.m_low = m_low
        self.m_high = m_high
        self.alpha = alpha
        self.beta = beta
        self.h = h
        self.name = name

        self.left_slope_fn, self.left_x_0 = point_slope_fn_factory(self.m_low, self.h, self.alpha, 'left')
        self.right_slope_fn, self.right_x_0 = point_slope_fn_factory(self.m_high, self.h, self.beta, 'right')

    def __add__(self, other):
        h = min(self.h, other.h)
        alpha = h * (self.alpha / self.h + other.alpha / other.h)
        beta = h * (self.beta / self.h + other.beta / other.h)
        m_low = self.m_low + other.m_low - self.alpha - other.alpha + alpha
        m_high = self.m_high + other.m_high + self.beta + other.beta - beta
        name = f"{self.name} + {other.name}"
        return TrapezoidFunciton(m_low, m_high, alpha, beta, h, name)

    def __mul__(self, other):
        return TrapezoidFunciton(self.m_low, self.m_high, self.alpha, self.beta, self.h * other,
                                 self.name + f"* {other}")

    def plot(self, x_lo, x_hi):
        x = [x_lo, self.left_x_0, self.m_low, self.m_high, self.right_x_0, x_hi]
        y = [0, 0, self.h, self.h, 0, 0]
        fig = plt.plot(x, y, label=self.name)
        plt.vlines([self.m_low, self.m_high], 0, self.h, linestyles='dashed', color='gray')
        plt.xticks(np.hstack([plt.xticks()[0], x]))
        plt.yticks(np.hstack([plt.yticks()[0], y]))
        plt.legend()

    def __call__(self, x):
        if x < self.left_x_0 or x > self.right_x_0:
            return 0
        elif self.left_x_0 <= x <= self.m_low:
            return self.left_slope_fn(x)
        elif self.m_low < x < self.m_high:
            return self.h
        elif self.m_high <= x <= self.right_x_0:
            return self.right_slope_fn(x)
        else:
            raise Exception("Something went wrong.")


def approx_union(s_list, x_lo, x_hi, d=1000):
    x = np.linspace(x_lo, x_hi, d)
    z = [list(map(fn, x)) for fn in s_list]
    y = list(map(max, zip(*z)))
    return x, y

A = TrapezoidFunciton(300 + 10 * i, 300 + 10 * i, 0, 0, 1, "A")
plt.yticks([0])
plt.xticks([300, 500])
A.plot(300, 500)

B = TrapezoidFunciton(300 + 5 * i, 350 + 10 * i, 300 + 5 * i - (250 + 10 * i), 400 + 10 * i - (350 + 10 * i), 1, "B")
plt.xticks([300, 500])
plt.yticks([0])
B.plot(300, 500)

C = TrapezoidFunciton(300, 300, 100, 0, 1, "C")
plt.xticks([100, 400])
plt.yticks([0])
C.plot(100, 400)

D_1 = TrapezoidFunciton(210, 270, 20, 30, 0.8, "D_1")
D_2 = TrapezoidFunciton(0, 0, 0, 0, 0.2, "D_2")
plt.xticks([200, 350])
plt.yticks([0])
D_1.plot(200, 350)
D_2.plot(200, 350)

E_1 = TrapezoidFunciton(0, 0, 0, 0, 0.8, "E_1")
E_2 = TrapezoidFunciton(300 + 5 * i, 300 + 5 * i, 0, 500 + 5 * i - (300 + 5 * i), 0.2, "E_2")
plt.xticks([0, 600])
plt.yticks([0])
E_1.plot(0, 600)
E_2.plot(0, 600)

s_list = [
    A + B + C + D_1 + E_1,
    A + B + C + D_1 + E_2,
    A + B + C + D_2 + E_1,
    A + B + C + D_2 + E_2
] # можливі варіанти фінансування

plt.xticks([1000, 2000])
plt.yticks([0])
for s in s_list:
    s.plot(1000, 2000)

x, y = approx_union(s_list, 1000, 2000)
plt.plot(x, y)

A1 = TrapezoidFunciton(100, 200, 30, 40, 1, "A_1")
A2 = TrapezoidFunciton(200, 300, 20, 60, 1, "A_2")
B1 = TrapezoidFunciton(140, 240, 30, 40, 1, "B_1")
B2 = TrapezoidFunciton(240, 320, 50, 40, 1, "B_2")
C1 = TrapezoidFunciton(50, 100, 10, 30, 1, "C_1")
C2 = TrapezoidFunciton(100, 150, 20, 50, 1, "C_2")

def larsen_method(A, B, C, x, y):
    alpha = min(A(x), B(y))
    C_hat = C * alpha
    return C_hat

Z1 = larsen_method(A1, B1, C1, 220, 200)
Z2 = larsen_method(A2, B2, C2, 220, 200)

plt.xticks([0, 250])
plt.yticks([0])
Z1.plot(0, 250)
Z2.plot(0, 250)

x, y = approx_union([Z1, Z2], 0, 250)
plt.plot(x, y)
