import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
processed_data_dir = os.path.join(data_dir, 'processed')
pdb_files_path = os.path.join(data_dir, 'pdbs')
uniprot_and_EC_path = os.path.join(processed_data_dir, '02_uniprotID_and_EC_reduced.csv')
protein_voxels_path = os.path.join(data_dir, 'protein_voxels')
dest = os.path.join(data_dir, 'datasets')

if not os.path.exists(dest):
    os.makedirs(dest)

if not os.path.exists(protein_voxels_path):
    os.makedirs(protein_voxels_path)

# open pdb file, save coordinates and atom types

def get_protein_voxels(pdb):
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

    max_x = max(xs) + 1
    max_y = max(ys) + 1
    max_z = max(zs) + 1

    # create the boxes
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

    atom_boxes = np.stack([carbon_box, oxygen_box, nitrogen_box, sulfur_box], axis=3)
    return atom_boxes

# Make EC numbers to class labels
uniprot_and_EC_data = pd.read_csv(uniprot_and_EC_path)
EC_numbers = uniprot_and_EC_data['EC'].to_list()
EC_numbers = [EC.split('.') for EC in EC_numbers]
for i, ECs in enumerate(EC_numbers):
    ECs = ECs[0:3]
    EC_numbers[i] = '.'.join(str(EC) for EC in ECs)

uniprot_and_EC_data['EC'] = EC_numbers

# example pdb file
pdb_files = [f for f in os.listdir(pdb_files_path) if f.endswith('pdb.gz')]
example_file_name = np.random.choice(pdb_files)
example_file = os.path.join(pdb_files_path, example_file_name)

# get atom boxes
protein_names = []

EC_numbers = []
pdb_files = [f for f in os.listdir(pdb_files_path) if f.endswith('pdb.gz')]
for file in pdb_files:

    protein_name = file.split('-')[1]
    file = os.path.join(pdb_files_path, file)

    EC_number = uniprot_and_EC_data[uniprot_and_EC_data['protein'] == protein_name]['EC'].values[0]
    protein_voxels = get_protein_voxels(file)

    protein_names.append(protein_name)
    EC_numbers.append(EC_number)
    with open(os.path.join(protein_voxels_path, f'{protein_name}.npy'), 'wb') as f:
        np.save(f, protein_voxels)



dataset = pd.DataFrame({
    'protein':protein_names,
    'EC':EC_numbers
})

print(dataset)
dataset.to_pickle(os.path.join(dest, '07_protein_voxelization_dataset.pkl'))

# plot
PLOT = False
if PLOT:
    colors = ['black', 'red', 'blue', 'yellow']
    for i in range(protein_voxels.shape[3]):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect('auto')
        ax.voxels(protein_voxels[:,:,:,i],
                  edgecolor="k",
                  facecolors=[colors[i]])
        plt.show(block=False)
        plt.pause(1)
        plt.close()


