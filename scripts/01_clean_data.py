import os
import pandas as pd

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
raw_data_dir = os.path.join(data_dir, 'raw')
raw_data_file = os.path.join(raw_data_dir, 'enzyme.dat')
dest = os.path.join(data_dir, 'processed')

if not os.path.exists(dest):
    os.makedirs(dest)

ID_and_uniprot = {}
with open(raw_data_file, 'r') as f:
    for line in f:
        if line.startswith('ID'):
            ID = line[5:].replace('\n', '')
            ID_and_uniprot[ID] = []

        if line.startswith('DR'):
            if len(line[5:11]) > 1:
                ID_and_uniprot[ID].append(line[5:11])
            if len(line[27:33]) > 1:
                ID_and_uniprot[ID].append(line[5:11])
            if len(line[49:55]) > 1:
                ID_and_uniprot[ID].append(line[5:11])

IDs = []
ECs = []

for EC in ID_and_uniprot:
    proteins = ID_and_uniprot[EC]
    if len(proteins) < 1:
        pass
    else:
        for protein in proteins:
            ECs.append(EC)
            IDs.append(protein)

data = pd.DataFrame({
    'protein': IDs,
    'EC': ECs
})
data = data.drop_duplicates()
data.to_csv(f'{dest}/01_uniprotID_and_EC.csv', index=None, sep=',')