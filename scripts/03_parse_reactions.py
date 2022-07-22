import os
import pandas as pd

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
raw_data_dir = os.path.join(data_dir, 'raw')
raw_data_file = os.path.join(raw_data_dir, 'enzyme.dat')
processed_data_path = os.path.join(data_dir, 'processed/01_uniprotID_and_EC.csv')
dest = os.path.join(data_dir, 'processed')


processed_data = pd.read_csv(processed_data_path)

unique_EC_numbers = pd.unique(processed_data.EC)

for EC in unique_EC_numbers:
    print(EC)
