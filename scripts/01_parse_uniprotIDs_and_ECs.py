import os
import pandas as pd
import re

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
            result_list = re.findall(r"\s\w+,", line)
            if result_list is not None: 
                result_list = "".join(result_list)
                result_list = re.sub(r" ", "", result_list, count = 0)
                result_list = result_list.split(",")
                for result in result_list:
                    if len(result) > 2: 
                        ID_and_uniprot[ID].append(result)
                result_list = list()

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
data.to_csv(f'{dest}/01_uniprotID_and_EC_raw.csv', index=None, sep=',')