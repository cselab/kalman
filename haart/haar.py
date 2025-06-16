import random
import matplotlib.pyplot as plt
import numpy as np
import math


def haar(t):
    h = 1, 1
    g = 1, -1
    f = len(h)
    l = len(t)
    y = [0] * l
    l2 = l // 2
    s = np.zeros(l2)
    d = np.zeros(l2)
    for j in range(l2):
        for k in range(f):
            s[j] += t[2 * j + k] * h[k] / 2
            d[j] += t[2 * j + k] * g[k] / 2
    return s, d


def conv(t, h):
    l2 = len(t)
    l = 2 * l2
    f = len(h)
    s = [0] * l
    for j in range(l2):
        for k in range(f):
            s[2 * j + k] = t[j] * h[k]
    return s


N = 110
x = -np.sin(2 * math.pi * np.arange(N) / N)
x1 = np.sin(2 * math.pi * np.arange(N) / N)
x[7 * N // 10:] = x1[7 * N // 10:]
s, d = haar(x)

plt.figure(figsize=(12, 6))
plt.axis([None, None, -1.1, 1.1])
plt.plot(x, '-k')
plt.savefig("haar.0.png", bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
plt.axis([None, None, -1.1, 1.1])
plt.step(x, '-k', where='post')
plt.savefig("haar.1.png", bbox_inches='tight')

plt.figure(figsize=(12, 6))
plt.axis([None, None, -1.1, 1.1])
plt.step(x, '-k', where='post')
plt.step(conv(s, [1, 1]), '-b', where='post')
plt.savefig("haar.2.png", bbox_inches='tight')

plt.figure(figsize=(12, 6))
plt.axis([None, None, -1.1, 1.1])
plt.step(x, '-k', where='post')
plt.step(conv(s, [1, 1]), '-b', where='post')
plt.step(conv(d, [1, -1]), '-r', where='post')
plt.savefig("haar.3.png", bbox_inches='tight')
plt.close()

i = np.argsort(np.abs(d))
n = len(i)
d[i[: 8 * n // 10]] = 0
x0 = np.add(conv(s, [1, 1]), conv(d, [1, -1]))
plt.figure(figsize=(12, 6))
plt.axis([None, None, -1.1, 1.1])
plt.step(x, '-k', where='post')
plt.step(conv(d, [1, -1]), '-r', where='post')
plt.step(x0, '-y', where='post')
plt.savefig("haar.4.png", bbox_inches='tight')
plt.close()
