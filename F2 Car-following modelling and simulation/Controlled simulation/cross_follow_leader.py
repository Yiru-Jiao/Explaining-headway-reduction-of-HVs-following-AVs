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
    speed_hat[0:3] = v_follower[0:3]
    position_hat = np.zeros_like(time) * np.nan
    position_hat[0:3] = x_follower[0:3]
    for t in np.arange(0,len(speed_hat)-3,1): # operational time interval is 0.3 second
        s_star[t] = s_0 + max(0., speed_hat[t]*T + speed_hat[t]*(speed_hat[t]-v_leader[t])/2/np.sqrt(alpha*beta))
        spacing_hat[t] = x_leader[t] - position_hat[t]# + l_leader/2 - l_follower/2
        if speed_hat[t]<=0. and spacing_hat[t]<s_0:
            acc_hat[t] = 0.
        else:
            acc_hat[t] = alpha * (1 - (speed_hat[t]/v_0)**delta - (s_star[t]/spacing_hat[t])**2)
        speed_hat[t+3] = max(0., speed_hat[t] + acc_hat[t] * (time[t+3]-time[t]))
        position_hat[t+3] = position_hat[t] + (speed_hat[t]+speed_hat[t+3])/2 * (time[t+3]-time[t])

    follower = pd.DataFrame({'time':time,'v_follower':speed_hat,'x_follower':position_hat})
    follower['l_follower'] = l_follower
    trajectory = follower.merge(leader, on='time', how='left')

    return trajectory


def gipps_estimation(para_gipps, leader, initial_follower, l_follower):
    v_0, s_0, tau, alpha, b, b_leader = para_gipps[['v_0', 's_0', 'tau', 'alpha', 'b', 'b_leader']].values
    id_tau = int(tau/0.1)

    time, x_leader, v_leader, l_leader = leader[['time','x_leader','v_leader','l_leader']].values.T
    l_leader = l_leader[0]
    x_follower, v_follower = initial_follower.values.T

    spacing_hat = np.zeros_like(time) * np.nan
    speed_hat = np.zeros_like(time) * np.nan
    speed_hat[:id_tau] = v_follower[:id_tau]
    position_hat = np.zeros_like(time) * np.nan
    position_hat[:id_tau] = x_follower[:id_tau]
    for t in np.arange(0,len(speed_hat)-id_tau,1):
        spacing_hat[t] = x_leader[t] - position_hat[t] #+ l_leader/2 - l_follower/2
        v_acc = speed_hat[t] + 2.5*alpha*tau*(1-speed_hat[t]/v_0) * np.sqrt(0.025+speed_hat[t]/v_0)
        braking_spacing = 2*(spacing_hat[t]-s_0)-tau*speed_hat[t]+(v_leader[t])**2/b_leader
        v_dec = -tau*b + np.sqrt(tau**2*b**2 + b*max(0., braking_spacing))
        speed_hat[t+id_tau] = max(0., min(v_acc, v_dec))
        position_hat[t+id_tau] = position_hat[t] + (speed_hat[t]+speed_hat[t+id_tau])/2 * (time[t+id_tau]-time[t])

    follower = pd.DataFrame({'time':time,'v_follower':speed_hat,'x_follower':position_hat})
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

    ## Make sure the start and end are not static, and the start is not free flow
    group_vleader = data_HH.groupby('case_id')['v_leader'].agg(['first','last'])
    group_vleader = group_vleader[(group_vleader['first']>1.)&
                                  (group_vleader['first']<10.)&
                                  (group_vleader['last']>1.)]
    group_vfollower = data_HH.groupby('case_id')['v_follower'].agg(['first','last'])
    group_vfollower = group_vfollower[(group_vfollower['first']>1.)&
                                      (group_vleader['first']<10.)&
                                      (group_vfollower['last']>1.)]
    group_non_static = group_vleader.index.intersection(group_vfollower.index)
    print('Number of filtered cases: {}'.format(len(group_non_static)))
    ## Make sure that more than 40% of the time is not in congestion
    group_high = data_HH[data_HH['v_leader']>10.].groupby('case_id')['time'].count()
    group_all = data_HH.groupby('case_id')['time'].count().loc[group_high.index]
    group_non_congested = group_high[group_high>0.4*group_all].index
    para_HH_lowerVar_id = regime_list_HH.loc[group_non_static.intersection(group_non_congested)].copy().drop(columns=['regime_comb'])
    print('Number of filtered cases: {}'.format(len(para_HH_lowerVar_id)))
    ## Sort the cases by duration of steady-state car-following, i.e., regime F
    para_HH_lowerVar_id = para_HH_lowerVar_id.sort_values(['F'], ascending=False)
    print(para_HH_lowerVar_id.head(4))

    # Repeat the test 4 times
    for count in range(4):
        para_HH_lowerVar = para_HH.loc[para_HH_lowerVar_id.index[count]]
        para_HH_higherVar = para_HH.loc[para_HH.index!=para_HH_lowerVar_id.index[count]]
        num_cases = len(para_HH_higherVar)

        follower_set = 'HH'
        follower_data = data_HH
        follower_idm = para_HH
        for leader_set in ['HHlowerVar','HHhigherVar']:
            trajectories = []
            case_id = 0

            follower_ids = np.random.RandomState(count).choice(para_HH_higherVar.index.unique(), size=num_cases, replace=False)
            if leader_set=='HHlowerVar':
                leader_ids = (np.ones(num_cases) * para_HH_lowerVar_id.index[count]).astype(int)
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
