import numpy as np
import matplotlib.pyplot as plt


def simple_cos_wave(t, f):
 return np.cos(2 * np.pi * f * t)

def simple_sin_wave(t, f):
 return np.sin(2 * np.pi * f * t)

f0 = 10
t = np.arange(0, 3 / f0, 1/(100 * f0))

plt.figure()
plt.plot(t, simple_cos_wave(t, f0), color = 'r', label='cos')
plt.plot(t, simple_sin_wave(t, f0), color = 'b', label='sin')
plt.legend()
plt.grid(linestyle=':')
plt.xlabel("time (sec)")
plt.ylabel("signals")
#plt.xlim([0, 100])
#plt.ylim([-20, 2])
plt.show()
#plt.savefig('toy.png', dpi=256)