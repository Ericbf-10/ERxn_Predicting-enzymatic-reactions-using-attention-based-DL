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
dest = os.path.join(data_dir, 'datasets')

if not os.path.exists(dest):
    os.makedirs(dest)

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
pdb_files = []
EC_numbers = []
pdb_file_names = [f for f in os.listdir(pdb_files_path) if f.endswith('pdb.gz')]
for file in pdb_file_names:
    try:
        protein_name = file.split('-')[1]
        file = os.path.join(pdb_files_path, file)
        EC_number = uniprot_and_EC_data[uniprot_and_EC_data['protein'] == protein_name]['EC'].values[0]
        pdb_files.append(file)
        EC_numbers.append(EC_number)
    except:
        pass


dataset = pd.DataFrame({
    'pdb':pdb_files,
    'EC':EC_numbers
})

print(dataset)
dataset.to_csv(os.path.join(dest, '07_protein_voxelization_dataset.csv'), index=False)

# # plot
# PLOT = False
# if PLOT:
#     colors = ['black', 'red', 'blue', 'yellow']
#     for i in range(protein_voxels.shape[3]):
#         fig = plt.figure()
#         ax = fig.gca(projection='3d')
#         ax.set_aspect('auto')
#         ax.voxels(protein_voxels[:,:,:,i],
#                   edgecolor="k",
#                   facecolors=[colors[i]])
#         plt.show(block=False)
#         plt.pause(1)
#         plt.close()


