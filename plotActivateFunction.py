#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 60)


def elu(x, a):
    y = []
    for i in x:
        if i >= 0:
            y.append(i)
        else:
            y.append(a * np.exp(i) - 1)
    return y


relu = np.maximum(x, [0] * 60)
relu6 = np.minimum(np.maximum(x, [0] * 60), [6] * 60)
softplus = np.log(np.exp(x) + 1)
elu = elu(x, 1)
softsign = x / (np.abs(x) + 1)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
lrelu = np.maximum(0.1 * x, x)
plt.figure()  # matplotlib 的 figure 就是一个 单独的 figure 小窗口
plt.ylim((-1.2, 1.2))

# plt.plot(x, relu6, label='relu6', linewidth=3.0)
plt.plot(x, relu, label='relu', color='black', linestyle='--', linewidth=2.0)
# plt.plot(x, elu, label='elu', linewidth=2.0)
# plt.plot(x, lrelu, label='lrelu', linewidth=1.0)
plt.plot(x, sigmoid, label='sigmoid', linewidth=2.0)
plt.plot(x, tanh, label='tanh', linewidth=2.0)
plt.title('Activate Function')

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
plt.legend(loc='best')
# plt.figure()
# plt.ylim((-1.2, 1.2))
# plt.plot(x, softsign, label='softsign', linewidth=2.0)
# plt.plot(x, sigmoid, label='sigmoid', linewidth=2.0)
# plt.plot(x, tanh, label='tanh', linewidth=2.0)
# plt.plot(x, softplus, label='softplus', linewidth=2.0)
# plt.plot(x, hyperbolic_tangent,label='hyperbolic_tangent',linewidth=2.0)
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0))
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))
# plt.legend(loc='best')
fig = plt.gcf()
fig.savefig('Activate Function.jpg')
plt.show()