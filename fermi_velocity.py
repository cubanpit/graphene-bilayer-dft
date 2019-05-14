'''Compute Fermi velocity in graphene from band structure JSON file.
'''


import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if len(sys.argv) != 2:
    raise RuntimeError('I want one and only one arguments: bandstructure.json')

bs_json = sys.argv[1]

with open(bs_json, 'r') as infile:
    bs = json.load(infile)

path = bs['path']
energies = np.array(bs['energies'][0])
ref = bs['reference']

kpts = path['kpts']
spec_kpts = path['special_points']
K_coord = spec_kpts['K']

cell = path['cell']['array']

# cell vector in metre
a1 = np.array(cell[0][:-1])

# norm of reciprocal lattice vector in metre^(-1)
b = 2 * np.pi / np.linalg.norm(a1)

Kidx = kpts.index(K_coord)

N = 12  # number of points to interpolate

idxN = np.where(energies[Kidx+N, :] > ref)[0][0]
jdxN = np.where(energies[Kidx+N, :] < ref)[0][-1]

top_band = idxN
bot_band = jdxN

K_right = range(Kidx, (Kidx + N))
K_left = range((Kidx - N + 1), (Kidx + 1))

# remove useless z compontent
kpts = np.array(kpts)[:, :-1] * b

hbar = 6.582e-16 * 1e10
fv = []

X, y = kpts[K_right], energies[K_right, top_band]
X = X - np.mean(X, axis=0)
y = y - np.mean(y)
slope, res, rank, s = np.linalg.lstsq(X, y, rcond=None)
fv.append(np.linalg.norm(slope) / hbar)
print('top-right', fv[-1])


X, y = kpts[K_left], energies[K_left, top_band]
X = X - np.mean(X, axis=0)
y = y - np.mean(y)
slope, res, rank, s = np.linalg.lstsq(X, y, rcond=None)
fv.append(np.linalg.norm(slope) / hbar)
print('top-left', fv[-1])

X, y = kpts[K_right], energies[K_right, bot_band]
X = X - np.mean(X, axis=0)
y = y - np.mean(y)
slope, res, rank, s = np.linalg.lstsq(X, y, rcond=None)
fv.append(np.linalg.norm(slope) / hbar)
print('bot-right', fv[-1])

X, y = kpts[K_left], energies[K_left, bot_band]
X = X - np.mean(X, axis=0)
y = y - np.mean(y)
slope, res, rank, s = np.linalg.lstsq(X, y, rcond=None)
fv.append(np.linalg.norm(slope) / hbar)
print('bot-left', fv[-1])

avg_fv = np.mean(fv)
err_fv = np.std(fv) / np.sqrt(3)
print('\nAvg', avg_fv)
ratio = avg_fv / 972895
print('\nratio', ratio, '\terr', ratio * err_fv / avg_fv)

# plt.plot(K_right, energies[K_right, top_band], label='top-right')
# plt.plot(K_right, energies[K_right, bot_band], label='bot-right')
# plt.plot(K_left, energies[K_left, top_band], label='top-left')
# plt.plot(K_left, energies[K_left, bot_band], label='bot-left')
# plt.legend()
# plt.show()
