from rdkit import Chem

from urllib.request import urlopen
from urllib.parse import quote

import pandas as pd
import os


def CIRconvert(ids):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans
    except:
        return 'Did not work'

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)



merged_reactions_and_smiles = pd.read_csv('../data/processed/05c_merged_reactions_and_smiles.csv')

tokenized_products = list()
tokenized_reactants = list()

product_list = list()
reactant_list = list()

reactant_product_dict = dict()

## ========================= ##
##          testing          ##
## ========================= ##


for i in range(merged_reactions_and_smiles.shape[0]):
    if merged_reactions_and_smiles['fully_covered'][i] == 1:
        reaction = merged_reactions_and_smiles['smiles'][i]
        print("Line: ", i)

        reactants = reaction.split(" = ")[0]
        products = reaction.split(" = ")[-1]

        r = list()
        for chemical in reactants.split(' + '):
            try: 
                c = Chem.MolToSmiles(Chem.MolFromSmiles(chemical), isomericSmiles = False)
            except: 
                c = 'Incomplete'
            r.append(c)
        untokenized_reactant = ''.join(r)
        
        p = list()
        for chemical in products.split(' + '):
            try: 
                c = Chem.MolToSmiles(Chem.MolFromSmiles(chemical), isomericSmiles = False)
            except: 
                c = 'Incomplete'
            p.append(c)
        untokenized_product = ''.join(p)

        if 'Incomplete' in untokenized_reactant or 'Incomplete' in untokenized_product:
            pass
        else: 
            try: 
                reactant = smi_tokenizer(untokenized_reactant)
                product = smi_tokenizer(untokenized_product)
                tokenized_reactants.append(reactant)
                tokenized_products.append(product)
            except:
                pass
            

print("Tokenized reactants: \n", tokenized_reactants[:5])
print("Tokenized products: \n", tokenized_products[:5])

reactants_products_df = pd.DataFrame({
    'reactants': tokenized_reactants, 
    'products': tokenized_products
})

reactants_products_df.to_csv('../data/processed/canonicalized_simles3.csv', index = False)

print(reactants_products_df.head(10))

random_seed = 100

train_df = reactants_products_df.sample(frac=0.7, random_state=random_seed)
tmp_df = reactants_products_df.drop(train_df.index)
test_df = tmp_df.sample(frac=0.33333, random_state=random_seed)
valid_df = tmp_df.drop(test_df.index)

assert len(reactants_products_df) == len(train_df) + len(valid_df) + len(test_df), "Dataset sizes don't add up"
del tmp_df


out_file_dir = '../data/datasets/mixed3/'

if not os.path.exists(out_file_dir):
    os.makedirs(out_file_dir)

test_df['reactants'].to_csv('../data/datasets/mixed3/src-test', index = False)
test_df['products'].to_csv('../data/datasets/mixed3/tgt-test', index = False)

train_df['reactants'].to_csv('../data/datasets/mixed3/src-train', index = False)
train_df['products'].to_csv('../data/datasets/mixed3/tgt-train', index = False)

valid_df['reactants'].to_csv('../data/datasets/mixed3/src-val', index = False)
valid_df['products'].to_csv('../data/datasets/mixed3/tgt-val', index = False)

