"""Band structure for graphene

Calculate the band structure of graphene bilayer in AB stacking
along special point in the BZ.
"""

import numpy as np
from ase import Atoms
from ase.visualize import view
from gpaw import GPAW, FermiDirac, PW
from ase.dft.kpoints import get_special_points

# experimental cell distance, bond length and layer distance
a = 2.46
b = a / np.sqrt(3)
d = 3.35

grap_bilayer = Atoms('C' * 4,
                 positions=[[0,     0, 0],
                            [0,     b, 0],
                            [0,     b, d],
                            [0, 2 * b, d]],
                 cell=[[      a,       0,     0],
                       [0.5 * a, 1.5 * b,     0],
                       [      0,       0, 10 * a]],
                 pbc=True)

# Perform standard ground state calculation (with plane wave basis)
# calc = GPAW(mode='lcao',
#             basis='dzp',
#             xc='PBE',
#             kpts=(10, 10, 1),
#             txt='graphene_bilayer_gs.txt')

calc = GPAW(mode=PW(400),
            xc='PBE',
            kpts=(10, 10, 1),
            random=True,  # random guess (needed if many empty bands required)
            occupations=FermiDirac(0.01),
            txt='graphene_bilayer_gs.txt')

grap_bilayer.calc = calc
en = grap_bilayer.get_potential_energy()
calc.write('graphene_bilayer_gs.gpw')
print('Potential Energy at first step:', en)

# Restart from ground state and fix potential:
calc = GPAW('graphene_bilayer_gs.gpw',
            nbands=25,
            fixdensity=True,
            symmetry='off',
            convergence={'bands': 16},
            kpts={'path': 'MKG', 'npoints': 200},
            txt='graphene_bilayer_gs.txt')

en = calc.get_potential_energy()
print('Potential Energy at second step:', en)

bs = calc.band_structure()
bs.write('bandstructure_AB.json')
