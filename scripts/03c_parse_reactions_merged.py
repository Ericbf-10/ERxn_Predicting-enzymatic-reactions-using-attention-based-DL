import os
import pandas as pd

# ============================================================== #
# Reduced the number of EC that has no protein or no fatas file, #
# ============================================================== #

file_location = '../data/fastas'

count = 0
protein_list = list()
for file in os.listdir(file_location):
    count += 1
    protein_list.append(file.split('.')[0])

print("Number of fastas file so far: ", count)

protein_EC_df = pd.read_csv('../data/processed/01a_brenda_uniprotID_and_EC_raw.csv')
protein_EC_dict = dict()

for i in range(len(protein_EC_df['protein'])):
    protein_EC_dict[protein_EC_df['protein'][i]] = protein_EC_df['EC'][i]

EC_list = list()

for protein in protein_list:
    EC_list.append(protein_EC_dict[protein])

brenda_reduced_protein_EC = pd.DataFrame({
    'protein': protein_list, 
    'EC':EC_list
})


brenda_reduced_protein_EC.to_csv('../data/processed/02b_brenda_uniprotID_and_EC_reduced.csv', index=None, sep=',')

EC_reaction_df = pd.read_csv("../data/processed/01b_brenda_reaction_and_EC_raw.csv")

before_EC_reaction_dict = dict()
for i in range(len(EC_reaction_df['EC'])):
    before_EC_reaction_dict[EC_reaction_df['EC'][i]] = EC_reaction_df['reaction'][i]

EC_key_list = list(before_EC_reaction_dict.keys())
reduced_EC_reaction_dict = dict()
removed_EC = list()

for ec in EC_key_list:
    if ec in EC_list:
        reduced_EC_reaction_dict[ec] = before_EC_reaction_dict[ec]
    else: 
        removed_EC.append(ec)

print("Before removing, number of EC in Brenda: ", len(EC_key_list))
print("After removing, number of EC in Brenda: ", len(reduced_EC_reaction_dict.keys()))


drop_index = list()

print("Number of EC that has no fasta file: ", len(removed_EC))
print("Number of unique EC in brenda: ", len(set(EC_list)))

brenda_reduced_EC_reaction_df = pd.DataFrame({
    'EC':list(reduced_EC_reaction_dict.keys()), 
    'reaction':list(reduced_EC_reaction_dict.values())
})


brenda_reduced_EC_reaction_df.to_csv('../data/processed/03b_brenda_ECs_and_reactions_reduced.csv', index=None, sep=',')


# ============================================================== #
# merged brenda and enzyme file by using both reduced EC file    #
# ============================================================== #

enzyme_reduced_EC_reaction_df = pd.read_csv('../data/processed/03_ECs_and_reactions.csv')

merged_EC_reaction_df = pd.concat([brenda_reduced_EC_reaction_df, enzyme_reduced_EC_reaction_df], axis = 0).drop_duplicates().reset_index(drop=True)
merged_EC_reaction_df.to_csv('../data/processed//03c_merged_EC_and_reaction_redcued.csv', index=None, sep=',')

