'''Write band structure to JSON file from GPAW file.
'''

import sys
import json
from gpaw import GPAW
from ase.parallel import parprint


def bs_to_json(band_structure, filename):
    '''Export bandstructure to JSON file.
    band_structure: an ase..dft.band_structure.BandStructure object
    filename: name of the output file
    '''

    # obtain useful information
    bsdict = band_structure.todict()
    bslab = band_structure.get_labels()

    # convert to list, otherwise it's not possible to export to JSON
    bsdict['energies'] = [list(array) for array in bsdict['energies'][0]]
    bsdict['kpts'] = list(bslab[0])
    bsdict['special_kpts'] = list(bslab[1])
    bsdict['special_labels'] = bslab[2]

    # delete 'path' information, not useful
    del bsdict['path']

    # actually write to file
    with open(filename, 'w') as fp:
        json.dump(bsdict, fp)


if len(sys.argv) != 2:
    raise RuntimeError('I want one and only one argument, the filename!')

filename = sys.argv[1]

calc = GPAW(filename, txt=None)
bs = calc.band_structure()
bs_to_json(bs, filename+'_bs.json')

parprint('Saved band structure file.')
