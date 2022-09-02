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
ec_data = ec_data[ec_data.groupby('EC')['EC'].transform('count')>=MIN_SAMPLES].copy()

point_clouds = []
EC_numbers = []
ECs = {}
i = 0
for index, row in ec_data.iterrows():
    if row[1] in ECs.keys():
        # if over 99 been sampled pass
        if ECs[row[1]] > 98:
            pass
        else:
            point_clouds.append(row[0]+'.txt')
            EC_numbers.append(row[1])
        ECs[row[1]] += 1

    # new EC number
    else:
        ECs[row[1]] = 0
        point_clouds.append(row[0] + '.txt')
        EC_numbers.append(row[1])

reduced_data = pd.DataFrame({
    'point_cloud':point_clouds,
    'EC':EC_numbers
})
print(reduced_data['EC'].value_counts())


reduced_data.to_csv(os.path.join(datasets_dir, '09_balanced_data_set.csv'), index=False)

# split data to train, test, validate
data = pd.read_csv(os.path.join(datasets_dir, '09_balanced_data_set.csv'))

# find number of enzymes per EC number
labels = data['EC'].unique()

train_x = []
train_y = []
test_x = []
test_y = []
val_x = []
val_y = []
for label in labels:
    df = data[data['EC'] == label]
    num_data = len(df)
    if num_data >= 20:
        valid = df.sample(frac=0.05, random_state=1)
        test = df.sample(frac=0.15, random_state=1)
        train = df.drop(test.index)
        train = df.drop(valid.index)
        for i, row in train.iterrows():
            train_x.append(row[0])
            train_y.append(row[1])
        for i, row in test.iterrows():
            test_x.append(row[0])
            test_y.append(row[1])
        for i, row in valid.iterrows():
            val_x.append(row[0])
            val_y.append(row[1])
    else:
        valid = df.sample(frac=0.1, random_state=1)
        test = df.sample(frac=0.2, random_state=1)
        train = df.drop(test.index)
        train = df.drop(valid.index)
        for i, row in train.iterrows():
            train_x.append(row[0])
            train_y.append(row[1])
        for i, row in test.iterrows():
            test_x.append(row[0])
            test_y.append(row[1])
        for i, row in valid.iterrows():
            val_x.append(row[0])
            val_y.append(row[1])


train = pd.DataFrame({
    'point_cloud': train_x,
    'EC': train_y
})

test = pd.DataFrame({
    'point_cloud': test_x,
    'EC': test_y
})

valid = pd.DataFrame({
    'point_cloud': val_x,
    'EC': val_y
})

train.to_csv(os.path.join(datasets_dir, '09_train.csv'), index=False)
test.to_csv(os.path.join(datasets_dir, '09_test.csv'), index=False)
valid.to_csv(os.path.join(datasets_dir, '09_valid.csv'), index=False)