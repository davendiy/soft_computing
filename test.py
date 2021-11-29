#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 2021-11-29
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

import softutils as sf

test = sf.sigmoid(sf.sin(sf.cos(sf.Var('x')) + sf.tan(sf.Var('y'))))
print(test)
print(test.partial_derivative('x'))
print(test.gradient())
