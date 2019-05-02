"""Band structure for graphene

Calculate the band structure of graphene along special point in the BZ
"""

import numpy as np
from ase import Atoms
from ase.visualize import view
from gpaw import GPAW, FermiDirac, PW
from ase.dft.kpoints import get_special_points

# experimental cell distance and bond length
a = 2.46
b = a / np.sqrt(3)

graphene = Atoms('C' * 2,
                 positions=[[0, 0, 0],
                            [0, b, 0]],
                 cell=[[    a,         0,      0],
                       [a / 2, 3 * b / 2,      0],
                       [                 0,         0, 5 * a]],
                 pbc=True)

# Perform standard ground state calculation (with plane wave basis)
calc = GPAW(mode='lcao',
            basis='dzp',
            xc='PBE',
            kpts=(10, 10, 1),
            txt='graphene_gs.txt')

# calc = GPAW(mode=PW(350),
#             xc='PBE',
#             kpts=(10, 10, 1),
#             random=True,  # random guess (needed if many empty bands required)
#             occupations=FermiDirac(0.01),
#             txt='graphene_gs.txt')

graphene.calc = calc
en = graphene.get_potential_energy()
calc.write('graphene_gs.gpw')
print('Potential Energy at first step:', en)

# Restart from ground state and fix potential:
calc = GPAW('graphene_gs.gpw',
            nbands=16,
            fixdensity=True,
            symmetry='off',
            kpts={'path': 'GMKG', 'npoints': 100},
            convergence={'bands': 8},
            txt='graphene_gs.txt')

en = calc.get_potential_energy()
print('Potential Energy at second step:', en)

bs = calc.band_structure()
bs.plot(filename='bandstructure.png', show=True, emin=-15, emax=15)
