import os
import pubchempy as pc
import pandas as pd
from tqdm import tqdm
import pickle

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
dest = os.path.join(data_dir, 'processed')
reactions_data = os.path.join(dest, '03_ECs_and_reactions.csv')
compounds_smile_data = os.path.join(dest, '04_compounds_smiles_mapper.csv')
synonyms_path = os.path.join(dest, '04_synonyms.pkl')
reactions_and_smiles = os.path.join(dest, '05_reactions_and_smiles.csv')

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

    for i in range(10):
        for j, compound in enumerate(compounds):
            if compound.startswith(f'{i}n '):
                compounds[j] = compound[len(f'{i}n '):]

    for i, compound in enumerate(compounds):
        if compound.startswith('n '):
            compounds[i] = compound[len('n '):]

    return compounds

with open(synonyms_path, 'rb') as f:
    synonyms = pickle.load(f)

reactions = pd.read_csv(reactions_data)

smile_data = pd.read_csv(compounds_smile_data)
compounds_with_smiles = smile_data['name'].to_list()

r_list = []
for i, reaction in enumerate(reactions['reaction'].to_list()):
    for synonym in synonyms.keys():
        reaction = reaction.replace(synonym, synonyms[synonym])
    r_list.append(reaction)

reactions = reactions.reset_index()
partially_covered_reactions = 0
fully_covered_reactions = 0
dropped_reactions = 0
remove_unknown = True
reactions_kept = []
fully_covered = []
for i, reaction in reactions.iterrows():
    reaction = reaction['reaction']
    try:
        compounds = split_reaction(reaction)
        n_compounds = len(compounds)
        count = 0
        for compound in compounds:
            if compound in synonyms.keys():
                reaction = reaction.replace(compound, synonyms[compound])

        for compound in compounds:
            if compound in compounds_with_smiles:
                reaction = reaction.replace(compound, smile_data.loc[smile_data['name'] == compound, 'smile'].item())
                count += 1
            else:
                if remove_unknown:
                    reaction = reaction.replace(compound, '')

        if remove_unknown == True:
            # case first reactant is removed
            if reaction.startswith(' + '):
                reaction = reaction[3:]

            # first product is removed
            if '  + ' in reaction:
                reaction = reaction.replace('  + ', ' ')

            # last educt is removed
            if ' +  ' in reaction:
                reaction = reaction.replace(' +  ', ' ')

            # intermediate product or educt is removed
            if '+  +' in reaction:
                reaction = reaction.replace('+  +', '+')

            # last product removed
            if reaction.endswith(' + '):
                reaction = reaction[:-3]

        if count == n_compounds:
            fully_covered_reactions += 1
            reactions_kept.append(reaction)
            fully_covered.append(1)
        else:
            partially_covered_reactions += 1
            reactions_kept.append(reaction)
            fully_covered.append(0)

    except:
        dropped_reactions +=1
        reactions.drop(index=i, axis=0, inplace=True)

reactions['smiles'] = reactions_kept
reactions['fully_covered'] = fully_covered

print('fully covered reactions:', fully_covered_reactions)
print('partially covered reactions:', partially_covered_reactions)
print('dropped reactions:', dropped_reactions)

reactions.to_csv(reactions_and_smiles, index=False)