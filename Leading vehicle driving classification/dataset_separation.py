'''
Separate Lyft data into train, val, and test sets.

Train set: 50% HA labeled 1 + 50% HH labeled 0
Val set: 50% HA labeled 1 + 50% HH labeled 0
Test set: 100% AH labeled 0

'''

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


parent_dir = os.path.abspath('..') # Set your parent directory here. 
                                   # Without change the current setting is the parent directory of this file.
original_data_path = parent_dir + 'Data path example/InputData/Lyft/CFdata/'
separated_data_path = parent_dir + 'Data path example/OutputData/AV leader/data/'
np.random.seed(131)


# Read data
def read_data(setname):
    data = pd.read_hdf(original_data_path+setname+'.h5', key='data')
    ## long enough cases
    data = data[data.groupby('case_id')['time'].transform('count')>=148]
    ## exclude cases with static states longer than 14.8s
    data['static'] = (data['v_follower']<0.1).astype(int)
    data['static'] = data.groupby('case_id')['static'].transform('sum')
    data = data[data['static']<148]
    return data[['case_id','time','v_leader','a_leader']]

data_AH = read_data('AH')
data_HA = read_data('HA')
data_HH = read_data('HH')


# Normalise features
dynamics_all = pd.concat([data_AH[['v_leader','a_leader']],
                          data_HA[['v_leader','a_leader']],
                          data_HH[['v_leader','a_leader']]])
max_v_leader = dynamics_all['v_leader'].max()
print('max v_leader:', max_v_leader)
dev_a_leader = np.sqrt((dynamics_all['a_leader']**2).sum()/len(dynamics_all))
print('deviation to zero a_leader:', dev_a_leader)

def normalise_data(data, max_v_leader, dev_a_leader):
    data['v_leader'] = data['v_leader']/max_v_leader
    data['a_leader'] = data['a_leader']/dev_a_leader
    return data

data_AH = normalise_data(data_AH, max_v_leader, dev_a_leader)
data_HA = normalise_data(data_HA, max_v_leader, dev_a_leader)
data_HH = normalise_data(data_HH, max_v_leader, dev_a_leader)


# Separate data into train, val, and test sets
case_ids_HA = data_HA['case_id'].unique()
case_ids_HH = data_HH['case_id'].unique()
case_ids_AH = data_AH['case_id'].unique()
num_train_HA = int(0.5*len(case_ids_HA))
num_train_HH = int(num_train_HA*1.094)

id2select_train_HA = np.random.choice(case_ids_HA, num_train_HA, replace=False)
id2select_val_HA = np.setdiff1d(case_ids_HA, id2select_train_HA)
num_val_HA = len(id2select_val_HA)
num_val_HH = int(num_val_HA*1.094)
id2select_train_HH = np.random.choice(case_ids_HH, num_train_HH, replace=False)
id2select_val_HH = np.random.choice(np.setdiff1d(case_ids_HH, id2select_train_HH), num_val_HH, replace=False)

data_train_HA = data_HA[data_HA['case_id'].isin(id2select_train_HA)]
data_val_HA = data_HA[data_HA['case_id'].isin(id2select_val_HA)]
data_train_HH = data_HH[data_HH['case_id'].isin(id2select_train_HH)]
data_val_HH = data_HH[data_HH['case_id'].isin(id2select_val_HH)]
data_test_AH = data_AH


# Rearange data into 14.8s segments, and assign labels
def rearange_data(data):
    data = data.sort_values(['case_id','time']).set_index('case_id')
    subcases = []
    new_id = 0
    for case_id in tqdm(data.index.unique()):
        duration = len(data.loc[case_id])
        num_subcases = duration//148 + 1
        for i in range(num_subcases-1):
            subcase = data.loc[case_id].iloc[i*148:(i+1)*148]
            subcase['subcase'] = i
            subcase['case_id'] = case_id
            subcase['new_id'] = new_id
            subcases.append(subcase)
            new_id += 1
        subcase = data.loc[case_id].iloc[-148:]
        subcase['subcase'] = num_subcases-1
        subcase['case_id'] = case_id
        subcase['new_id'] = new_id
        subcases.append(subcase)
        new_id += 1
    data = pd.concat(subcases).reset_index(drop=True)
    return data

data_train_HA = rearange_data(data_train_HA)
data_train_HA['new_id'] = (data_train_HA['new_id'] + 1e6).astype(int)
data_train_HA['label'] = 1
label_train_HA = data_train_HA.groupby('new_id').agg({'case_id':'first','subcase':'first'}).reset_index()
label_train_HA['label'] = 1

data_val_HA = rearange_data(data_val_HA)
data_val_HA['new_id'] = (data_val_HA['new_id'] + 1e6).astype(int)
data_val_HA['label'] = 1
label_val_HA = data_val_HA.groupby('new_id').agg({'case_id':'first','subcase':'first'}).reset_index()
label_val_HA['label'] = 1

data_train_HH = rearange_data(data_train_HH)
data_train_HH['new_id'] = (data_train_HH['new_id'] + 3e6).astype(int)
data_train_HH['label'] = 0
label_train_HH = data_train_HH.groupby('new_id').agg({'case_id':'first','subcase':'first'}).reset_index()
label_train_HH['label'] = 0

data_val_HH = rearange_data(data_val_HH)
data_val_HH['new_id'] = (data_val_HH['new_id'] + 3e6).astype(int)
data_val_HH['label'] = 0
label_val_HH = data_val_HH.groupby('new_id').agg({'case_id':'first','subcase':'first'}).reset_index()
label_val_HH['label'] = 0

data_test_AH = rearange_data(data_test_AH)
data_test_AH['new_id'] = data_test_AH['new_id'].astype(int)
data_test_AH['label'] = 0
label_test_AH = data_test_AH.groupby('new_id').agg({'case_id':'first','subcase':'first'}).reset_index()
label_test_AH['label'] = 0

data_train_HAandHH = pd.concat([data_train_HA, data_train_HH]).reset_index(drop=True)
label_train_HAandHH = pd.concat([label_train_HA, label_train_HH]).reset_index(drop=True)
data_val_HAandHH = pd.concat([data_val_HA, data_val_HH]).reset_index(drop=True)
label_val_HAandHH = pd.concat([label_val_HA, label_val_HH]).reset_index(drop=True)
data_test_AHonly = data_test_AH.reset_index(drop=True)
label_test_AHonly = label_test_AH.reset_index(drop=True)

print('train_HA new_id:', data_train_HA['new_id'].nunique(), 'train_HA case_id:', data_train_HA['case_id'].nunique())
print('val_HA new_id:', data_val_HA['new_id'].nunique(), 'val_HA case_id:', data_val_HA['case_id'].nunique())
print('train_HH new_id:', data_train_HH['new_id'].nunique(), 'train_HH case_id:', data_train_HH['case_id'].nunique())
print('val_HH new_id:', data_val_HH['new_id'].nunique(), 'val_HH case_id:', data_val_HH['case_id'].nunique())
print('test_AH new_id:', data_test_AH['new_id'].nunique(), 'test_AH case_id:', data_test_AH['case_id'].nunique())


print('Saving data...')
data_train_HAandHH.to_hdf(separated_data_path+'data_train_HAandHH.h5', key='data', mode='w')
label_train_HAandHH.to_hdf(separated_data_path+'label_train_HAandHH.h5', key='label', mode='w')
data_val_HAandHH.to_hdf(separated_data_path+'data_val_HAandHH.h5', key='data', mode='w')
label_val_HAandHH.to_hdf(separated_data_path+'label_val_HAandHH.h5', key='label', mode='w')
data_test_AHonly.to_hdf(separated_data_path+'data_test_AHonly.h5', key='data', mode='w')
label_test_AHonly.to_hdf(separated_data_path+'label_test_AHonly.h5', key='label', mode='w')
