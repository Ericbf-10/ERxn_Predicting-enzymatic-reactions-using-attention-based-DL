import tarfile
import os
from tqdm import tqdm
import pandas as pd

# Alphafold external DB
EXTERNAL_DRIVE = '/Volumes/Extreme SSD/Swiss_Prot'
compressed_AF_files = os.path.join(EXTERNAL_DRIVE, 'swissprot_pdb_v3.tar')


# if not downloaded, download:
if not os.path.exists(compressed_AF_files):
    AF_DB_SWISS_PROT = 'https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v3.tar'
    os.system(f'wget {AF_DB_SWISS_PROT} -P {EXTERNAL_DRIVE}')


tarf = tarfile.open(compressed_AF_files)

# general paths
script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
processed_data_dir = os.path.join(data_dir, 'processed')
uniprot_files = os.path.join(processed_data_dir, '02_uniprotID_and_EC_reduced.csv')
dest = os.path.join(data_dir, 'pdbs')


if not os.path.exists(dest):
    os.makedirs(dest)

uniprot_data = pd.read_csv(uniprot_files)
uniprotIDs = uniprot_data['protein'].to_list()


with tarfile.open(compressed_AF_files) as tarf:
    for member in tqdm(tarf.getmembers()):
        if member.name.split('-')[1] in uniprotIDs:
            tarf.extract(member, dest)

print(os.listdir(dest))
