'''Estimate Fermi velocity in graphene from band structure JSON file.

The usual bandstructure is on a MKG path in the Brillouin zone, in this script
we take the four branches around the K point, compute the slope through a
linear fit fot each branch and then average the four slopes to get a single
value. This is a very rough estimate.
'''


import sys
import json
import numpy as np


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

# 2D cell vectors in angstrom
a1 = np.array(cell[0][:-1])

# norm of reciprocal lattice vector in angstrom^(-1)
b = 2 * np.pi / np.linalg.norm(a1)

# Number of points to interpolate.
# Increase this if the slope is not realistic, but keep it low to stay in the
#  linear and low energy region, based how dense the sampling in the BZ is.
N = 8

# index of K point in kpts array
Kidx = kpts.index(K_coord)

# index of top and bottom bands around the K point, in the energies array
top_band = np.where(energies[Kidx+N, :] > ref)[0][0]
bot_band = np.where(energies[Kidx+N, :] < ref)[0][-1]

# right and left branches in the kpts array
K_right = range(Kidx, (Kidx + N))
K_left = range((Kidx - N + 1), (Kidx + 1))

# remove useless z compontent
kpts = np.array(kpts)[:, :-1] * b

# physical constant, rescaled because we are working with angstroms
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

# Ratio with single layer graphene Fermi velocity computed with the same method.
# Update if the method changes.
ratio = avg_fv / 972895
print('\nratio', ratio, '\terr', ratio * err_fv / avg_fv)

