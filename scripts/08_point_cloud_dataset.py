import os
import gzip
import numpy as np
import pandas as pd

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
processed_data_dir = os.path.join(data_dir, 'processed')
pdb_files_path = os.path.join(data_dir, 'pdbs')
uniprot_and_EC_path = os.path.join(processed_data_dir, '02_uniprotID_and_EC_reduced.csv')
dataset_dir = os.path.join(data_dir, 'datasets')
dest = os.path.join(data_dir, 'point_cloud_dataset')

MAX_LENGTH = 10000

def get_point_cloud(pdb_file):
    """
    read a pdb file, extract atom coordinates and type and save them as Mx4x2 vector

    Mx1:3x1 are the atom coordinates Mx3:4x1 is padding
    Mx4x2 are atom types
    """
    xs = []
    ys = []
    zs = []
    atom_types = []
    atoms = ['C', 'O', 'N', 'S', 'CA']
    with gzip.open(pdb_file, 'rb') as f:
        for line in f:
            if line.startswith(b'ATOM'):
                line = line.decode()
                xs.append(float(line[30:38])) # x
                ys.append(float(line[38:46])) # y
                zs.append(float(line[46:54])) # z
                if line[13] == atoms[0]: atom_types.append(0)
                if line[13] == atoms[1]: atom_types.append(1)
                if line[13] == atoms[2]: atom_types.append(2)
                if line[13] == atoms[3]: atom_types.append(3)

    M = len(xs)
    point_cloud = np.zeros((len(xs),7))
    for i in range(M):
        atom_type_index = atom_types[i]
        point_cloud[i, 0] = xs[i]
        point_cloud[i, 1] = ys[i]
        point_cloud[i, 2] = zs[i]
        point_cloud[i, atom_type_index+3] = 1

    return point_cloud



if not os.path.exists(dest):
    os.makedirs(dest)

# Make EC numbers to class labels
uniprot_and_EC_data = pd.read_csv(uniprot_and_EC_path)
EC_numbers = uniprot_and_EC_data['EC'].to_list()

uniprot_and_EC_data['EC'] = EC_numbers

# get point clouds
point_clouds = []
EC_numbers = []
pdb_file_names = [f for f in os.listdir(pdb_files_path) if f.endswith('pdb.gz')]
for file in pdb_file_names:
    try:
        protein_name = file.split('-')[1]
        file = os.path.join(pdb_files_path, file)
        EC_number = uniprot_and_EC_data[uniprot_and_EC_data['protein'] == protein_name]['EC'].values[0]
        point_cloud = get_point_cloud(file)
        point_cloud_path = protein_name + '.txt'
        if point_cloud.shape[0] <= MAX_LENGTH:
            point_clouds.append(point_cloud_path)
            EC_numbers.append(EC_number)
            np.savetxt(os.path.join(dest, point_cloud_path), point_cloud, fmt='%1.3f')

    except:
        pass


dataset = pd.DataFrame({
    'point_cloud':point_clouds,
    'EC':EC_numbers
})

print(dataset)
dataset.to_csv(os.path.join(dataset_dir, '08_point_cloud_dataset.csv'), index=False)