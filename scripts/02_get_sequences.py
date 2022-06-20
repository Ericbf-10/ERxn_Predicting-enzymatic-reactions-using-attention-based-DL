import requests as r
from Bio import SeqIO
from io import StringIO
import os
import pandas as pd
from tqdm import tqdm

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
processed_data_dir = os.path.join(data_dir, 'processed')
uniprot_files = os.path.join(processed_data_dir, '01_uniprot_and_EC.csv')
dest = os.path.join(data_dir, 'fastas')

def get_fasta(cID, dest):
    baseUrl="http://www.uniprot.org/uniprot/"
    currentUrl=baseUrl+cID+".fasta"
    response = r.post(currentUrl)
    cData=''.join(response.text)

    Seq=StringIO(cData)
    pSeq=list(SeqIO.parse(Seq,'fasta'))
    fasta = pSeq[0].format('fasta')

    file_name = os.path.join(dest, cID + '.fasta')

    with open(file_name, 'w') as f:
        seq = ''
        lines = fasta.split('\n')
        for i, line in enumerate(lines):
            if i == 0:
                pass
            else:
                seq = seq + line
        f.write(seq)

data = pd.read_csv(uniprot_files)
cIDs = data.protein.to_list()

downloaded = os.listdir(dest)
downloaded = [file[:-6] for file in downloaded]

for i in range(100):
    for cID in tqdm(cIDs):
        if cID not in downloaded:
            try:
                get_fasta(cID, dest)
            except:
                pass