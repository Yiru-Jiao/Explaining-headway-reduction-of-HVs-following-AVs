'''
Let the same followers follow leaders with different variety.

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
        spacing_hat[t] = x_leader[t] - position_hat[t] #+ l_leader/2 - l_follower/2
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


for model, simulate in zip(['idm','gipps'], [idm_estimation, gipps_estimation]):
    print('Model: {}'.format(model))

    para_HH = pd.read_csv(data_path+model+'/parameters_Lyft_HH.csv', index_col=0).dropna()
    data_HH = pd.read_hdf(data_path+'cfdata_idm_Lyft_HH.h5', key='data').sort_values(['case_id','time']).set_index('case_id')
    regime_list_HH = pd.read_csv(parent_dir+'Data/OutputData/CF regime/Lyft/regimes/regimes_list_HH.csv', index_col=0)

    # Repeat the test 5 times
    for count in tqdm(range(5)):
        para_HH_lowerVar_id = regime_list_HH.loc[para_HH.index].sort_values(['F','S'], ascending=False).index[count]
        para_HH_lowerVar = para_HH.loc[para_HH_lowerVar_id]
        para_HH_higherVar = para_HH.loc[para_HH.index!=para_HH_lowerVar_id]
        num_cases = len(para_HH_higherVar)

        follower_set = 'HH'
        follower_data = data_HH
        follower_idm = para_HH
        for leader_set in ['HHlowerVar','HHhigherVar']:
            trajectories = []
            case_id = 0

            follower_ids = np.random.RandomState(count).choice(para_HH_higherVar.index.unique(), size=num_cases, replace=False)
            if leader_set=='HHlowerVar':
                leader_ids = (np.ones(num_cases) * para_HH_lowerVar_id).astype(int)
                leader_data = data_HH
            elif leader_set=='HHhigherVar':
                leader_ids = np.random.RandomState(5+count).choice(para_HH_higherVar.index.unique(), size=num_cases, replace=False)
                leader_data = data_HH

            for leader_id, follower_id in zip(leader_ids,follower_ids):
                leader = leader_data.loc[leader_id]
                initial_follower = leader[['x_follower','v_follower']]
                leader = leader[['time','a_leader','x_leader','v_leader','l_leader']]

                para_idm = follower_idm.loc[follower_id]
                length_follower = follower_data.loc[follower_id]['l_follower'].iloc[0]

                traj = simulate(para_idm, leader, initial_follower, length_follower)
                traj['leader_id'] = leader_id
                traj['follower_id'] = follower_id
                traj['case_id'] = case_id
                case_id += 1
                trajectories.append(traj)
                
            trajectories = pd.concat(trajectories).reset_index(drop=True)
            trajectories = trajectories[trajectories['leader_id']!=trajectories['follower_id']]
            trajectories = trajectories.drop_duplicates(['leader_id','follower_id','time'])
            print('follower: {}, leader: {}, number of cases: {}'.format(follower_set, leader_set, len(trajectories['case_id'].unique())))
            trajectories['dhw'] = trajectories['x_leader'] - trajectories['x_follower']# - trajectories['l_follower']/2 + trajectories['l_leader']/2
            trajectories['thw'] = trajectories['dhw']/trajectories['v_follower']

            trajectories.to_hdf(data_path+'crossfollow/'+model+'/crossfollow_Lyft_f'+follower_set+'_l'+leader_set+'_'+str(count)+'.h5', key='data', mode='w')
