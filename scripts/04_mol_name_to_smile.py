import os
import pubchempy as pc
import pandas as pd
import time
from tqdm import tqdm


script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
dest = os.path.join(data_dir, 'processed')
reactions_data = os.path.join(dest, '03_ECs_and_reactions.csv')
compounds_smile_data = os.path.join(dest, '04_compounds_smiles_mapper.csv')

# get all compound names
reactions = pd.read_csv(reactions_data)['reaction']

def split_reaction(reaction):

    educts = reaction.split(' = ')[0]
    products = reaction.split(' = ')[1]
    educts = educts.split(' + ')
    products = products.split(' + ')
    compounds = educts + products

    # handeling stoichiometry
    for i in range(20):
        for j, compound in enumerate(compounds):
            if compound.startswith(f'{i} '):
                compounds[j] = compound[len(f'{i} '):]

    for i, compound in enumerate(compounds):
        if compound.startswith('n '):
            compounds[i] = compound[len('n '):]

    return compounds

def get_smiles(compounds, accepted_compounds, accepted_smiles, rejected_compounds):
    for compound in compounds:
        if compound in accepted_compounds or compound in rejected_compounds:
            pass
        else:
            try:
                time.sleep(0.1)
                result = pc.get_properties(properties='IsomericSMILES', identifier=compound, namespace='name')[0]
                smile = result['IsomericSMILES']

                accepted_compounds.append(compound)
                accepted_smiles.append(smile)
            except:
                rejected_compounds.append(compound)

    return accepted_compounds, accepted_smiles, rejected_compounds

i = 0
accepted_compounds = []
accepted_smiles = []
rejected_compounds = []

for reaction in tqdm(reactions):
    try:
        compounds = split_reaction(reaction)
        accepted_compounds, accepted_smiles, rejected_compounds = get_smiles(compounds, accepted_compounds, accepted_smiles, rejected_compounds)
    except:
        pass

accepted = pd.DataFrame({
    'name':accepted_compounds,
    'smile':accepted_smiles
})

accepted.to_csv(compounds_smile_data)

with open('rejected_compounds.txt', 'w') as f:
    for compound in rejected_compounds:
        f.writelines(compound + '\n')
