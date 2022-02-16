#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 2021-11-29
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

import softutils as sf

test = sf.relu(sf.sigmoid(sf.sin(sf.cos(sf.Var('x')) + sf.tan(sf.Var('y')))))
print(test)
print(test.partial_derivative('x'))
print(test.gradient())
print(test.partial_derivative('x').partial_derivative('x'))


test2 = sf.sin(sf.Var('x') ** 2 + sf.Var('y') ** 2)
print(test2.minimize('x', eta=0.01, y=1))
print(test2.maximize('x', eta=0.01, y=1))
