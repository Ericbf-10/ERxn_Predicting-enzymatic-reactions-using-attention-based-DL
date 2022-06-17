import os

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
raw_data_dir = os.path.join(data_dir, 'raw')

expasy_url = 'https://ftp.expasy.org/databases/enzyme/enzyme.dat'
os.system(f'wget {expasy_url} -P {raw_data_dir}')