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

    for i in range(10):
        for j, compound in enumerate(compounds):
            if compound.startswith(f'{i}n '):
                compounds[j] = compound[len(f'{i}n '):]

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
                result = pc.get_properties(properties='IsomericSMILES', identifier=compound, namespace='name')[0]
                smile = result['IsomericSMILES']
                accepted_compounds.append(compound)
                accepted_smiles.append(smile)
            except:
                rejected_compounds.append(compound)

    return accepted_compounds, accepted_smiles, rejected_compounds

def parenthetic_contents(string):
    """Generate parenthesized contents in string as pairs (level, contents)."""
    stack = []
    for i, c in enumerate(string):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            start = stack.pop()
            yield (len(stack), string[start + 1: i], (start+1,i))

def get_synonym(compound):
    # try to get synonyms for some compounds
    for i in range(20):
        if f'{i}-alpha' in compound:
            compound = compound.replace(f'{i}-alpha', f'{i}alpha')

    for i in range(20):
        if f'{i}-beta' in compound:
            compound = compound.replace(f'{i}-beta', f'{i}beta')

    return compound

def reformat_brackets(compound):
    results = list(parenthetic_contents(compound))

    max_ = 0
    for i in results:
        if max_ < i[0]:
            max_ = i[0]

    parentheses = ['()','[]','{}']
    brackets = []
    j = 0
    for _ in range(max_+1):
        brackets.append(parentheses[j])
        if j == 2:
            j = 0
        else:
            j += 1

    brackets.reverse()

    for i, r in enumerate(results):
        indices = r[2]
        compound = compound[:indices[0]-1] + brackets[r[0]][0] + compound[indices[0]:indices[1]] + brackets[r[0]][1] + compound[indices[1]+1:]
    return compound


def filter_compounds(compounds, synonyms):

    for i, comp in enumerate(compounds):
        if comp in synonyms.keys():
            compounds[i] = synonyms[comp]
    # skip things that wont be found, because they are macromolecules
    compounds = [x for x in compounds if 'DNA' not in x]
    compounds = [x for x in compounds if 'RNA' not in x]
    compounds = [x for x in compounds if 'Side' not in x]
    compounds = [x for x in compounds if 'protein' not in x.lower()]
    compounds = [x for x in compounds if 'transfer' not in x.lower()]
    compounds = [x for x in compounds if 'synthase' not in x.lower()]
    compounds = [x for x in compounds if 'channel' not in x.lower()]
    compounds = [x for x in compounds if 'complex' not in x.lower()]
    compounds = [x for x in compounds if 'carrier' not in x.lower()]
    compounds = [x for x in compounds if 'chain' not in x.lower()]
    compounds = [x for x in compounds if 'tubulin' not in x.lower()]
    compounds = [x for x in compounds if 'ferredoxin' not in x.lower()]
    compounds = [x for x in compounds if 'phospholipid' not in x.lower()]
    compounds = [x for x in compounds if 'globulin' not in x.lower()]
    compounds = [x for x in compounds if 'fatty' not in x.lower()]
    compounds = [x for x in compounds if 'procollagen' not in x.lower()]
    compounds = [x for x in compounds if 'histone' not in x.lower()]
    compounds = [x for x in compounds if 'reductase' not in x.lower()]
    compounds = [x for x in compounds if 'siderophore' not in x.lower()]
    compounds = [x for x in compounds if 'elongation' not in x.lower()]
    compounds = [x for x in compounds if 'carboxylase' not in x.lower()]
    compounds = [x for x in compounds if 'aldolase' not in x.lower()]
    compounds = [x for x in compounds if 'cypemycin' not in x.lower()]
    compounds = [x for x in compounds if 'polymer' not in x.lower()]
    compounds = [x for x in compounds if 'system' not in x.lower()]
    compounds = [x for x in compounds if 'synthetase' not in x.lower()]
    compounds = [x for x in compounds if 'receptor' not in x.lower()]

    # skip vague descriptions
    compounds = [x for x in compounds if '(n)' not in x]
    compounds = [x for x in compounds if '(n+1)' not in x]
    compounds = [x for x in compounds if '(m)' not in x]
    compounds = [x for x in compounds if 'reaction' not in x.lower()]
    compounds = [x for x in compounds if 'other' not in x.lower()]
    compounds = [x for x in compounds if 'subunit' not in x.lower()]
    compounds = [x for x in compounds if 'product' not in x.lower()]
    compounds = [x for x in compounds if 'group' not in x.lower()]
    compounds = [x for x in compounds if 'donor' not in x.lower()]
    compounds = [x for x in compounds if 'acceptor' not in x.lower()]
    compounds = [x for x in compounds if not x.lower().startswith('a ')]
    compounds = [x for x in compounds if not x.lower().startswith('an ')]
    return compounds


# speedup in case script has already been run before - don't check compounds that have already been found
if os.path.exists(compounds_smile_data):
    data = pd.read_csv(compounds_smile_data)
    accepted_compounds = data['name'].to_list()
    accepted_smiles = data['smile'].to_list()
    with open(os.path.join(dest, '04_unmapped_compounds.txt'), 'r') as f:
        rejected_compounds = f.readlines()
    skip_basic_search = True
else:
    accepted_compounds = []
    accepted_smiles = []
    rejected_compounds = []
    skip_basic_search = False


# synonyms for some common compounds
synonyms = {
    'NAD(P)(+)': 'NADP(+)',
    'NAD(P)H': 'NADPH',
    'H(2)O': 'water',
    'CO(2)': 'CO2',
    'H(2)O(2)': 'hydrogen peroxide',
    'FADH(2)': 'FADH2',
    'HCO(3)(-)': 'hydrogen carbonate',
    'H(2)S': 'sulfane',
    'N(2)': 'N2',
    'H(2)': 'H2',
    'NAD(P)+': 'NADP(+)',
    'NAD(H)': 'NADH',
    'NMN(H)': 'NMNH',
    'H(2)CO(3)': 'H2CO3',
    'O(2)':'O2',
    'aflatoxin B(1)':'aflatoxin B1',
    'aflatoxin B(2)':'aflatoxin B2'
    }

# main loop looking for compounds
if skip_basic_search == False:
    print('primary search loop')
    for reaction in tqdm(reactions):
        try:
            compounds = split_reaction(reaction)
            compounds = filter_compounds(compounds, synonyms)
            accepted_compounds, accepted_smiles, rejected_compounds = get_smiles(compounds, accepted_compounds, accepted_smiles, rejected_compounds)
        except:
            pass


# catch alpha beta spellings
print('Searching for different syntax for alpha and beta')
for rejected_compound in tqdm(rejected_compounds):
    if rejected_compound in synonyms.keys():
        synonym = synonyms[rejected_compound]
    else:
        synonym = get_synonym(rejected_compound)

    if synonym == rejected_compound:
        pass
    else:
        try:
            result = pc.get_properties(properties='IsomericSMILES', identifier=synonym, namespace='name')[0]
            smile = result['IsomericSMILES']
            accepted_compounds.append(synonym)
            accepted_smiles.append(smile)
            synonyms[rejected_compound] = synonym
        except:
            rejected_compounds.remove(rejected_compound)

# catch nested brackets
print('Searching for nested brackets')
for rejected_compound in tqdm(rejected_compounds):
    if rejected_compound in synonyms.keys():
        synonym = synonyms[rejected_compound]
    else:
        synonym = reformat_brackets(rejected_compound)

    if synonym == rejected_compound:
        pass
    else:
        try:
            result = pc.get_properties(properties='IsomericSMILES', identifier=synonym, namespace='name')[0]
            smile = result['IsomericSMILES']
            accepted_compounds.append(synonym)
            accepted_smiles.append(smile)
            synonyms[rejected_compound] = synonym
        except:
            rejected_compounds.remove(rejected_compound)

accepted = pd.DataFrame({
    'name':accepted_compounds,
    'smile':accepted_smiles
})

accepted.to_csv(compounds_smile_data, index=False)

print(synonyms)
with open(os.path.join(dest, '04_unmapped_compounds.txt'), 'w') as f:
    for compound in rejected_compounds:
        if compound.endswith('\n'):
            f.writelines(compound)
        else:
            f.writelines(compound + '\n')

with open(os.path.join(dest, '04_synonyms.pkl'), 'wb') as f:
    pickle.dump(synonyms, f)
