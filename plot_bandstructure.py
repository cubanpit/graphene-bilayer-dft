'''Import band structure from JSON file and plot it.
'''

import sys
import json
import matplotlib.pyplot as plt


def pretty(kpt):
    if kpt == 'G':
        kpt = r'$\Gamma$'
    elif len(kpt) == 2:
        kpt = kpt[0] + '$_' + kpt[1] + '$'
    return kpt


if len(sys.argv) != 2:
    raise RuntimeError('I want one and only one argument, the filename!')

filename = sys.argv[1]

with open(filename, 'r') as fp:
    bsdict = json.load(fp)

ereference = bsdict['reference']
energies = bsdict['energies']
kpts = bsdict['kpts']
special_kpts = bsdict['special_kpts']
special_labs = [pretty(name) for name in bsdict['special_labels']]

emin = -3
emax = +3
emin += ereference
emax += ereference

ax = plt.figure().add_subplot(111)
ax.axis(xmin=0, xmax=kpts[-1], ymin=emin, ymax=emax)
ax.set_xticks(special_kpts)
ax.set_xticklabels(special_labs)
ax.axhline(ereference, color='k', ls=':')

plt.plot(kpts, energies, color='black')

plt.show()
