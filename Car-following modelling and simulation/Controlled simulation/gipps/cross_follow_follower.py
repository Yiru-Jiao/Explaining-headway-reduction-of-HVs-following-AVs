'''
Let followers with different variety follow the same leaders.

'''

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')



def idm_estimation(para_idm, leader, initial_follower, l_follower):
    v_0, s_0, T, alpha, beta = para_idm[['v_0', 's_0', 'T', 'alpha', 'beta']].values
    delta = 4.
    time, x_leader, v_leader, l_leader = leader[['time','x_leader','v_leader','l_leader']].values.T
    l_leader = l_leader[0]
    x_follower, v_follower = initial_follower.values.T

    s_star = np.zeros_like(time) * np.nan
    acc_hat = np.zeros_like(time) * np.nan
    spacing_hat = np.zeros_like(time) * np.nan
    speed_hat = np.zeros_like(time) * np.nan
    speed_hat[0] = v_follower[0]
    position_hat = np.zeros_like(time) * np.nan
    position_hat[0] = x_follower[0]
    for t in np.arange(0,len(speed_hat)-1,1): # operational time interval is 0.3 second
        s_star[t] = s_0 + max(0., speed_hat[t]*T + speed_hat[t]*(speed_hat[t]-v_leader[t])/2/np.sqrt(alpha*beta))
        spacing_hat[t] = x_leader[t] - position_hat[t] + l_leader/2 - l_follower/2
        if speed_hat[t]<=0. and spacing_hat[t]<s_0:
            acc_hat[t] = 0.
        else:
            acc_hat[t] = alpha * (1 - (speed_hat[t]/v_0)**delta - (s_star[t]/spacing_hat[t])**2)
        speed_hat[t+1] = speed_hat[t] + acc_hat[t] * (time[t+1]-time[t])
        speed_hat[speed_hat<0.] = 0.
        position_hat[t+1] = position_hat[t] + (speed_hat[t]+speed_hat[t+1])/2 * (time[t+1]-time[t])
    acc_hat[t+1] = acc_hat[t]

    follower = pd.DataFrame({'time':time,'a_follower':acc_hat,'v_follower':speed_hat,'x_follower':position_hat})
    follower['l_follower'] = l_follower
    trajectory = follower.merge(leader, on='time', how='left')

    return trajectory


parent_dir = os.path.abspath('..') # Set your parent directory here. 
                                   # Without change the current setting is the parent directory of this file.
data_path = parent_dir + 'Data path example/OutputData/IDM/'

idm_HH = pd.read_csv(data_path+'parameters_idm_Lyft_HH.csv', index_col=0).dropna()
data_HH = pd.read_hdf(data_path+'cfdata_idm_Lyft_HH.h5', key='data').sort_values(['case_id','time']).set_index('case_id')

idm_HH_lowerVar = idm_HH.loc[np.random.RandomState(100).choice(idm_HH.index, size=291, replace=False)]
idm_HH_higherVar = idm_HH.loc[~idm_HH.index.isin(idm_HH_lowerVar.index)]
idm_HH_lowerVar.to_csv(data_path+'parameters_idm_Lyft_HHlowerVar.csv')
idm_HH_higherVar.to_csv(data_path+'parameters_idm_Lyft_HHhigherVar.csv')


follower_data = data_HH
for follower_set, leader_set in zip(['HHlowerVar','HHhigherVar'], ['HH','HH']):
    trajectories = []
    case_id = 0

    for repeat in tqdm(range(121)):
        if follower_set=='HHlowerVar':
            if repeat%10 > 0:
                continue
            num_cases = len(idm_HH_lowerVar)
            follower_ids = np.random.RandomState(200+repeat).choice(idm_HH_lowerVar.index, size=num_cases, replace=False)
            follower_idm = idm_HH_lowerVar
            leader_ids = np.random.RandomState(300+repeat).choice(idm_HH.index, size=num_cases, replace=False)
            leader_data = data_HH
        elif follower_set=='HHhigherVar':
            if repeat%41 > 0:
                continue
            num_cases = len(idm_HH_higherVar)
            follower_ids = np.random.RandomState(400+repeat).choice(idm_HH_higherVar.index, size=num_cases, replace=False)
            follower_idm = idm_HH_higherVar
            leader_ids = np.random.RandomState(500+repeat).choice(idm_HH.index, size=num_cases, replace=False)
            leader_data = data_HH

        for leader_id, follower_id in zip(leader_ids,follower_ids):
            leader = leader_data.loc[leader_id]
            initial_follower = leader[['x_follower','v_follower']]
            leader = leader[['time','a_leader','x_leader','v_leader','l_leader']]

            para_idm = follower_idm.loc[follower_id]
            length_follower = follower_data.loc[follower_id]['l_follower'].iloc[0]

            traj = idm_estimation(para_idm, leader, initial_follower, length_follower)
            traj['leader_id'] = leader_id
            traj['follower_id'] = follower_id
            traj['case_id'] = case_id
            case_id += 1
            trajectories.append(traj)
            
    trajectories = pd.concat(trajectories).reset_index(drop=True)
    trajectories = trajectories[trajectories['leader_id']!=trajectories['follower_id']]
    trajectories = trajectories.drop_duplicates(['leader_id','follower_id','time'])
    print('follower: {}, leader: {}, number of cases: {}'.format(follower_set, leader_set, len(trajectories['case_id'].unique())))
    trajectories['dhw'] = trajectories['x_leader'] - trajectories['x_follower'] - trajectories['l_follower']/2 + trajectories['l_leader']/2
    trajectories['thw'] = trajectories['dhw']/trajectories['v_follower']

    trajectories.to_hdf(data_path+'crossfollow/crossfollow_Lyft_f'+follower_set+'_l'+leader_set+'.h5', key='data')
