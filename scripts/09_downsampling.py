import os
import matplotlib.pyplot as plt
import sys
import pandas as pd
sys.path.append('functions/')

MAX_SAMPLES = 100
MIN_SAMPLES = 10

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
processed_data_dir = os.path.join(data_dir, 'processed')
pdb_files_path = os.path.join(data_dir, 'pdbs')
point_cloud_path = os.path.join(data_dir, 'point_cloud_dataset')
datasets_dir = os.path.join(data_dir, 'datasets')

# open data sets
pc_data_path = os.path.join(data_dir, 'datasets/08_point_cloud_dataset.csv')
ec_data_path = os.path.join(data_dir, 'processed/02_uniprotID_and_EC_reduced.csv')
pc_data = pd.read_csv(pc_data_path)
ec_data = pd.read_csv(ec_data_path)

# find number of enzymes per EC number
ec_nums = ec_data['EC'].to_list()

ec_data['EC'] = ec_nums

# filter low_counts
ec_data = ec_data[ec_data.groupby('EC')['EC'].transform('count')>MIN_SAMPLES].copy()
ec_data = ec_data[ec_data.groupby('EC')['EC'].transform('count')<MAX_SAMPLES].copy()

down_sampled_ECs = ec_data['EC'].to_list()
pc_data = pc_data[pc_data['EC'].isin(down_sampled_ECs)]

pc_data.to_csv(os.path.join(datasets_dir, '09_balanced_data_set.csv'), index=False)