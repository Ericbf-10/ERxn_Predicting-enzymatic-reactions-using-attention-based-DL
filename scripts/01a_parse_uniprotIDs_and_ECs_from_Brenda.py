import os
import pandas as pd
import re


script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '..', 'data')

raw_data_dir = os.path.join(data_dir, 'raw')
raw_data_file = os.path.join(raw_data_dir, 'brenda_2022_2.txt')
dest = os.path.join(data_dir, 'processed')

if not os.path.exists(dest):
    os.makedirs(dest)


ID_and_uniprot = {}
ID_flag = False
ID_and_reaction = {}
uniprot_results = list()
PR_flag = False
RE_flag = False
reaction = ""
reaction_list = list()

with open(raw_data_file, 'r') as f:
    for line in f:
        ID_result = re.findall('^ID\s([0-9\.]+)', line)
        if ID_result is not None and len(ID_result) != 0 :
                ID_flag = True
                uniprot_results = list()
                ID = ID_result[0]
                ID_and_uniprot[ID] = []
                ID_and_reaction[ID] = []

        if ID_flag and (line.startswith("PR") or PR_flag):
            PR_flag = True
            uniprot_result = re.findall('\s([A-Z0-9]+)\sUni[p|P]rot', line)
            if len(uniprot_result) != 0: 
                uniprot_results.append(uniprot_result)

        if ID_flag and line.startswith("RECOMMENDED_NAME"): 
            PR_flag = False
            proteins = list()
            for protein in uniprot_results: 
                proteins += protein
            ID_and_uniprot[ID] = proteins
            
        if ID_flag and (line.startswith("RE\t") or RE_flag):
            RE_flag = True
            line = line.replace("RE", "//")
            line = line.replace("\t", " ")
            reaction += line.replace("\n", " ")

        if RE_flag and line.startswith("\n"):
            RE_flag = False
            if len(reaction) != 0:
                reaction_list = reaction.split("//")
                reaction_list.remove("")
                ID_and_reaction[ID] = reaction_list
                reaction_list = list()
                reaction = ""


IDs = []
ECs = []
reactions = []

for EC in ID_and_uniprot:
    proteins = ID_and_uniprot[EC]
    if len(proteins) < 1:
        pass
    else:
        for protein in proteins:
            ECs.append(EC)
            IDs.append(protein)

uniprot_data = pd.DataFrame({
    'protein': IDs,
    'EC': ECs
})

IDs = []

for EC in ID_and_reaction:
    reaction = ID_and_reaction[EC]

    if len(reaction) < 1:
        pass
    else:
        for r in reaction:
            IDs.append(EC)
            reactions.append(r)

reaction_data = pd.DataFrame({
    'EC': IDs,
    'reaction': reactions
})


uniprot_data = uniprot_data.drop_duplicates()
reaction_data = reaction_data.drop_duplicates()

uniprot_data.to_csv(f'{dest}/01a_brenda_uniprotID_and_EC_raw.csv', index=None, sep=',')
reaction_data.to_csv(f'{dest}/01a_brenda_reaction_and_EC_raw.csv', index=None, sep=',')
