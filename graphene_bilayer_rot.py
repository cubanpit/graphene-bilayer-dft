"""Band structure for graphene

Calculate the band structure of graphene along special point in the BZ
"""

import numpy as np
from ase import Atoms
from ase.visualize import view
from gpaw import GPAW
from ase.dft.kpoints import get_special_points, bandpath
from ase.parallel import parprint
import matplotlib.pyplot as plt
import sys


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

        >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.math.atan2(np.cross(v1_u, v2_u), np.dot(v1_u, v2_u))


def repeat_cell(pos, cell, size):
    '''Repeat unit cell of graphene in order to obtain
       a larger hexagonal cell, with a side of length 'size'.

       At the moment works based on this structure:

             #     #
          #     #     #

          #     #     #
       A     #     #     #

       B     #     #     #
          #     #     #

          #     #     #
             #     #
    '''

    # positions of big unit cell
    big_pos = np.array(pos)
    cell = np.array(cell)

    # build the first two sides
    for i in range(2*size - 3):
        if i == 0 or i % 2 == 1:
            big_pos = np.append(big_pos,
                                (big_pos[i*2] + cell[1]).reshape(1, -1),
                                axis=0)
            big_pos = np.append(big_pos,
                                (big_pos[i*2 + 1] + cell[1]).reshape(1, -1),
                                axis=0)
        if i == 0 or i % 2 == 0:
            big_pos = np.append(big_pos,
                                (big_pos[i*2]
                                    + cell[0] - cell[1]).reshape(1, -1),
                                axis=0)
            big_pos = np.append(big_pos,
                                (big_pos[i*2 + 1]
                                    + cell[0] - cell[1]).reshape(1, -1),
                                axis=0)

    # build everything up to the end of the top/bottom side
    for i in range(2 * (size - 1) * (2*size - 1)):
        big_pos = np.append(big_pos,
                            (big_pos[i] + cell[0]).reshape(1, -1),
                            axis=0)

    # complete the last part up to the corner
    last = 2 * (size - 1) * (2*size - 1)
    side = 2 * (2*size - 1)
    for i in range(size - 1):
        for j in range(last, last + side - 4):
            big_pos = np.append(big_pos,
                                (big_pos[j] + cell[0]).reshape(1, -1),
                                axis=0)
        last = last + side
        side = side - 4

    return big_pos


# experimental cell distance, bond length and layer distance
a = 2.46
b = a / np.sqrt(3)
d = 3.35

# indices for rotation -> moving from (n, m) to (m, n)
# n, m = 2, 1       # 21.79 deg
# n, m = 4, 3       # 9.43 deg
# n, m = 13, 12     # 2.65 deg
n, m = 32, 31     # 1.05 deg
gcd = np.gcd(n, m)
n /= gcd
m /= gcd

# unrotated layer
un_el_pos = np.array([[0, 0],
                      [0, b]])
un_el_cell = np.array([[a,       0],
                       [0.5 * a, 1.5 * b]])
un_pos = repeat_cell(un_el_pos, un_el_cell, int(n))

# relative rotation angle (RRA) and 2D rotation matrix
v1 = m * un_el_cell[0] + n * un_el_cell[1]
v2 = n * un_el_cell[0] + m * un_el_cell[1]
theta = angle_between(v2, v1)
c, s = np.cos(theta), np.sin(theta)
R = np.array([[c, -s], [s, c]])
parprint('# RRA =', np.degrees(theta))

# rotated layer from AB stacking
rot_pos = np.array([x + np.array([-0.5*a, 0.5*b]) for x in un_pos])
rot_pos = np.array([R @ x for x in rot_pos])

# supercell positions
un_pos = np.hstack((un_pos, 0 * np.ones(len(un_pos)).reshape(-1, 1)))
rot_pos = np.hstack((rot_pos, d * np.ones(len(rot_pos)).reshape(-1, 1)))
super_pos = np.vstack((un_pos, rot_pos))
expected_number = 4 * (n**2 + m**2 + n*m)
if len(super_pos) != expected_number:
    raise RuntimeError("Number of atoms in the supercell doesn't match"
                       " theoretical prediction, there is something wrong!")

# change recursion limit, otherwise ase.Atoms will reach it and crash
if expected_number > 1000:
    sys.setrecursionlimit(int(expected_number*1.1))

# supercell vectors
v3 = -m * un_el_cell[0] + (n + m) * un_el_cell[1]
v2 = R @ v2
v3 = R @ v3
v_norm = np.linalg.norm(v2)

# If "height" of the cell is less then the "side" ase.kpoints will crash, so
# we set this temporary value for vacuum, then we correct it, it doesn't really
# matter because we are working on a bandpath on the x-y plane.
vacuum_tmp = v_norm
super_cell = np.array([np.append(v2, 0),
                       np.append(v3, 0),
                       [0, 0, vacuum_tmp]])

grap_bilayer = Atoms('C' * len(super_pos),
                     positions=super_pos,
                     cell=super_cell,
                     pbc=True)

# build path in BZ for bandstructure calculation
points = get_special_points(super_cell, lattice='hexagonal')
GMKG = [points[k] for k in 'GMKG']
kpts, x, X = bandpath(GMKG, super_cell, 200)

# now we set the real value for vacuum
vacuum = 15
super_cell[2][2] = vacuum
grap_bilayer = Atoms('C' * len(super_pos),
                     positions=super_pos,
                     cell=super_cell,
                     pbc=True)

# Perform standard ground state calculation (with plane wave basis)
calc = GPAW(mode='lcao',
            basis='sz(dzp)',
            nbands='110%',
            xc='PBE',
            kpts=(5, 5, 1),
            txt='graphene_bilayer_gs.txt',
            parallel=dict(band=2,              # band parallelization
                          augment_grids=True,  # use all cores for XC/Poisson
                          sl_auto=True)        # enable parallel ScaLAPACK
           )

grap_bilayer.calc = calc
en1 = grap_bilayer.get_potential_energy()
calc.write('graphene_bilayer_gs.gpw')

# Restart from ground state and fix potential:
calc = GPAW('graphene_bilayer_gs.gpw',
            nbands='110%',
            fixdensity=True,
            symmetry='off',
            kpts=kpts,
            # convergence={'bands': '110%'},
            txt='graphene_bilayer_gs.txt',
            parallel=dict(band=2,              # band parallelization
                          augment_grids=True,  # use all cores for XC/Poisson
                          sl_auto=True)        # enable parallel ScaLAPACK
           )

en2 = calc.get_potential_energy()
parprint('Energy self-consistent:', en1, '\nEnergy bandstructure:', en2)

bs = calc.band_structure()
bs.write(filename='bandstructure_rot'+
         str(np.round(np.degrees(theta),2))+'.json')
