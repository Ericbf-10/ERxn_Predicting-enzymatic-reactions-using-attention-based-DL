import os
import gzip
import time

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
processed_data_dir = os.path.join(data_dir, 'processed')
pdb_files_path = os.path.join(data_dir, 'pdbs')
dest = os.path.join(data_dir, 'micro_environments')

if not os.path.exists(dest):
    os.makedirs(dest)

# open pdb file, save coordinates and atom types

def get_atoms(pdb, MICRO_BOX_SIZE=5):
    '''
    takes a pdb file and saves the atom coordinates of carbon, oxygen,
    nitrogen and sulfur atoms as stacked 3 dimensional np arrays
    '''
    xs = []
    ys = []
    zs = []
    atom_types = []
    CAs = []
    atoms = ['C', 'O', 'N', 'S', 'CA']
    with gzip.open(pdb, 'rb') as f:
        i = 0
        for line in f:
            if line.startswith(b'ATOM'):
                line = line.decode()
                xs.append(round(float(line[30:38]))) # x
                ys.append(round(float(line[38:46]))) # y
                zs.append(round(float(line[46:54]))) # z
                if line[13] == atoms[0]: atom_types.append(0)
                if line[13] == atoms[1]: atom_types.append(1)
                if line[13] == atoms[2]: atom_types.append(2)
                if line[13] == atoms[3]: atom_types.append(3)
                if line[13:15] == atoms[4]: CAs.append(i)
                i += 1

    # move the origin, so there are no negative coordinates
    xs = [x - min(xs) for x in xs]
    ys = [y - min(ys) for y in ys]
    zs = [z - min(zs) for z in zs]

    # create a 3d box
    MICRO_BOX_SIZE = MICRO_BOX_SIZE

    max_x = max(xs)
    max_y = max(ys)
    max_z = max(zs)

    while max_x % MICRO_BOX_SIZE != 0:
        max_x += 1

    while max_y % MICRO_BOX_SIZE != 0:
        max_y += 1

    while max_z % MICRO_BOX_SIZE != 0:
        max_z += 1

    max_x += 1
    max_y += 1
    max_z += 1

    carbon_box = np.zeros((max_x, max_y, max_z), dtype=int)
    oxygen_box = np.zeros((max_x, max_y, max_z), dtype=int)
    nitrogen_box = np.zeros((max_x, max_y, max_z), dtype=int)
    sulfur_box = np.zeros((max_x, max_y, max_z), dtype=int)

    for i, atom_type in enumerate(atom_types):
        if atom_type == 0:
            carbon_box[xs[i], ys[i], zs[i]] = 1
        if atom_type == 1:
            oxygen_box[xs[i], ys[i], zs[i]] = 1
        if atom_type == 2:
            nitrogen_box[xs[i], ys[i], zs[i]] = 1
        if atom_type == 3:
            sulfur_box[xs[i], ys[i], zs[i]] = 1

    atom_boxes = np.stack([carbon_box, oxygen_box, nitrogen_box, sulfur_box])
    return atom_boxes


def micro_environment_type1(atom_boxes):
    '''
    Takes atom boxes and returns a sequence of micro environments by deviding the atom box
    into equally sized micro environments
    '''
    carbon_box = atom_boxes[0]
    oxygen_box = atom_boxes[1]
    nitrogen_box = atom_boxes[2]
    sulfur_box = atom_boxes[3]

    grid_x = [x * 5 for x in range(carbon_box.shape[0] // MICRO_BOX_SIZE)]
    grid_y = [y * 5 for y in range(carbon_box.shape[1] // MICRO_BOX_SIZE)]
    grid_z = [z * 5 for z in range(carbon_box.shape[2] // MICRO_BOX_SIZE)]
    micro_env_sequence = []
    for x in grid_x:
        for y in grid_y:
            for z in grid_z:
                cbox = carbon_box[x:x + MICRO_BOX_SIZE, y:y + MICRO_BOX_SIZE, z:z + MICRO_BOX_SIZE]
                obox = oxygen_box[x:x + MICRO_BOX_SIZE, y:y + MICRO_BOX_SIZE, z:z + MICRO_BOX_SIZE]
                nbox = nitrogen_box[x:x + MICRO_BOX_SIZE, y:y + MICRO_BOX_SIZE, z:z + MICRO_BOX_SIZE]
                sbox = sulfur_box[x:x + MICRO_BOX_SIZE, y:y + MICRO_BOX_SIZE, z:z + MICRO_BOX_SIZE]
                micro_env_sequence.append(np.stack([cbox, obox, nbox, sbox]))

    return micro_env_sequence

# example file
pdb_files = [f for f in os.listdir(pdb_files_path) if f.endswith('pdb.gz')]
example_file_name = np.random.choice(pdb_files)
example_file = os.path.join(pdb_files_path, example_file_name)
MICRO_BOX_SIZE = 5

# get atom boxes
atom_boxes = get_atoms(example_file, MICRO_BOX_SIZE=5)

# devide 3d box into sequence of cubes with dimension MICRO_BOX_SIZE x MICRO_BOX_SIZE x MICRO_BOX_SIZE
micro_env_sequence = micro_environment_type1(atom_boxes)

for i, box in enumerate(micro_env_sequence):
    Cbox = box[0]
    if np.all((Cbox == 0)):
        pass
    else:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect('auto')
        ax.voxels(Cbox, edgecolor="k")
        plt.title(f'{i}/{len(micro_env_sequence)}')
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()

# here will be the atom types
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# protein = ax.scatter(xs=carbon_box[0,:],
#                      ys=carbon_box[1,:],
#                      zs=carbon_box[2,:])#,
#                      #c=atom_types)
#
# ax.legend(handles=protein.legend_elements()[0],
#           title='atoms')
#
# plt.title(example_file_name)
#
# plt.show()
