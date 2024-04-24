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
    for t in np.arange(0,len(speed_hat)-1,1): # update every 0.1 second
        s_star[t] = s_0 + max(0., speed_hat[t]*T + speed_hat[t]*(speed_hat[t]-v_leader[t])/2/np.sqrt(alpha*beta))
        spacing_hat[t] = x_leader[t] - position_hat[t]# + l_leader/2 - l_follower/2
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


def gipps_estimation(para_gipps, leader, initial_follower, l_follower):
    v_0, s_0, tau, alpha, b, b_leader = para_gipps[['v_0', 's_0', 'tau', 'alpha', 'b', 'b_leader']].values
    theta = tau/2
    id_tau = int(tau/0.1)

    time, x_leader, v_leader, l_leader = leader[['time','x_leader','v_leader','l_leader']].values.T
    l_leader = l_leader[0]
    x_follower, v_follower = initial_follower.values.T

    spacing_hat = np.zeros_like(time) * np.nan
    speed_hat = np.zeros_like(time) * np.nan
    speed_hat[:id_tau] = v_follower[:id_tau]
    position_hat = np.zeros_like(time) * np.nan
    position_hat[:id_tau] = x_follower[:id_tau]
    for t in np.arange(0,len(speed_hat)-id_tau,1): # update every 0.1 second
        spacing_hat[t] = x_leader[t] - position_hat[t]
        v_acc = speed_hat[t] + 2.5*alpha*tau*(1-speed_hat[t]/v_0) * np.sqrt(0.025+speed_hat[t]/v_0)
        v_dec = -(tau+theta)*b + np.sqrt((tau+theta)**2*b**2 + b*(2*(spacing_hat[t]-s_0)-tau*speed_hat[t]+(v_leader[t])**2/b_leader))
        speed_hat[t+id_tau] = max(0., min(v_acc, v_dec))
        position_hat[t+id_tau] = position_hat[t+id_tau-1] + (speed_hat[t+id_tau-1]+speed_hat[t+id_tau])/2 * (time[t+id_tau]-time[t+id_tau-1])

    follower = pd.DataFrame({'time':time[id_tau:],'v_follower':speed_hat[id_tau:],'x_follower':position_hat[id_tau:]})
    follower['l_follower'] = l_follower
    trajectory = follower.merge(leader, on='time', how='left')

    return trajectory



parent_dir = './' # Set your parent directory here. 
                                   # Without change the current setting is the parent directory of this file.
data_path = parent_dir + 'Data/OutputData/Variability/'


for model in ['idm','gipps']:
    para_HH = pd.read_csv(data_path+model+'/parameters_idm_Lyft_HH.csv', index_col=0).dropna()
    data_HH = pd.read_hdf(data_path+'cfdata_idm_Lyft_HH.h5', key='data').sort_values(['case_id','time']).set_index('case_id')

    para_HH_lowerVar = para_HH.loc[np.random.RandomState(100).choice(para_HH.index, size=291, replace=False)]
    para_HH_higherVar = para_HH.loc[~para_HH.index.isin(para_HH_lowerVar.index)]
    para_HH_lowerVar.to_csv(data_path+'crossfollow/'+model+'/parameters_idm_Lyft_HHlowerVar.csv')
    para_HH_higherVar.to_csv(data_path+'crossfollow/'+model+'/parameters_idm_Lyft_HHhigherVar.csv')

    follower_data = data_HH
    for follower_set, leader_set in zip(['HHlowerVar','HHhigherVar'], ['HH','HH']):
        trajectories = []
        case_id = 0

        for repeat in tqdm(range(121)):
            if follower_set=='HHlowerVar':
                if repeat%10 > 0:
                    continue
                num_cases = len(para_HH_lowerVar)
                follower_ids = np.random.RandomState(200+repeat).choice(para_HH_lowerVar.index, size=num_cases, replace=False)
                follower_idm = para_HH_lowerVar
                leader_ids = np.random.RandomState(300+repeat).choice(para_HH.index, size=num_cases, replace=False)
                leader_data = data_HH
            elif follower_set=='HHhigherVar':
                if repeat%41 > 0:
                    continue
                num_cases = len(para_HH_higherVar)
                follower_ids = np.random.RandomState(400+repeat).choice(para_HH_higherVar.index, size=num_cases, replace=False)
                follower_idm = para_HH_higherVar
                leader_ids = np.random.RandomState(500+repeat).choice(para_HH.index, size=num_cases, replace=False)
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
