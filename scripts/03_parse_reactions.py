import os
import pandas as pd
from joblib import Parallel, delayed

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
raw_data_dir = os.path.join(data_dir, 'raw')
raw_data_file = os.path.join(raw_data_dir, 'enzyme.dat')
processed_data_dir = os.path.join(data_dir, 'processed')
uniprot_and_EC_data_path = os.path.join(data_dir, 'processed/02_uniprotID_and_EC_reduced.csv')
dest = os.path.join(data_dir, 'processed')

def get_reactions(EC, file):
    reactions = []
    with open(file, 'r') as f:
        record = False
        for line in f:
            if line == 'ID   ' + EC + '\n':
                record =True
            if line.startswith('CA') and record == True:
                reactions.append(line)
            if line.startswith('//') and record == True:
               break

    reactions = clean_reactions(reactions)

    return reactions


def clean_reactions(reactions):

    # simple cas: only one reaction
    if len(reactions) == 1:
        reaction = reactions[0][5:-2]
        return([reaction])

    # multiple reactions
    all_reactions = []
    if len(reactions) > 1:
        merged_reactions = ''.join(reactions).replace('\n', '').split('.')[:-1]
        for reaction in merged_reactions:
            for i in range(1,7):
                reaction = reaction.replace(f'CA   ({i}) ', '')
            reaction = reaction.replace('+CA   ', '+ ')
            reaction = reaction.replace('CA   ', ' ')
            reaction = reaction.replace('  ', ' ')
            reaction = reaction.replace('- ', '-')
            if reaction.startswith(' '):
                reaction = reaction[1:]
            all_reactions.append(reaction)

    return all_reactions

data = pd.read_csv(uniprot_and_EC_data_path)
unique_EC_numbers = pd.unique(data.EC)

# parse all reactions using parallelization (handle cores and parallelization with argparse in the future)
all_reactions = Parallel(n_jobs=8)(delayed(get_reactions)(EC, raw_data_file) for EC in unique_EC_numbers)

ECs = []
rxns = []
for i, reactions in enumerate(all_reactions):
    for reaction in reactions:
        ECs.append(unique_EC_numbers[i])
        rxns.append(reaction)


data = pd.DataFrame({
    'EC':ECs,
    'reaction':rxns
})

data.to_csv(f'{processed_data_dir}/03_ECs_and_reactions.csv', index=None, sep=',')