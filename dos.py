'''Write dos to JSON and plot dos, starting from GPAW calculator file.
'''

import sys
from gpaw import GPAW
from ase.dft.dos import DOS
import matplotlib.pyplot as plt


if len(sys.argv) != 2:
    raise RuntimeError('I want one and only one argument, the filename!')

filename = sys.argv[1]

calc = GPAW(filename, txt=None)

dos = DOS(calc, width=0.01, npts=1001)

d = dos.get_dos()
e = dos.get_energies()

plt.plot(e, d)
plt.xlabel('energy [eV]')
plt.ylabel('DOS')
plt.show()
