'''Write DOS to JSON file, starting from GPAW calculator file.
'''

import sys
import json
from gpaw import GPAW
from ase.dft.dos import DOS


if len(sys.argv) != 2:
    raise RuntimeError('I want one and only one argument, the filename!')

filename = sys.argv[1]

calc = GPAW(filename, txt=None)

dos = DOS(calc, width=0.1)

dos_dict = {}
dos_dict['dos'] = list(dos.get_dos())
dos_dict['energies'] = list(dos.get_energies())

out_name = filename + '_dos.json'
with open(out_name, 'w') as outfile:
    json.dump(dos_dict, outfile)

print('DOS written to', out_name)
