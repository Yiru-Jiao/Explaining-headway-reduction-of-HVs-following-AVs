'''
Compute calibration loss of IDM.

'''

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def idm_loss(cfdata,parameters):
    position, speed, acceleration, time = cfdata[['x_follower','v_follower','a_follower','time']].values.T

    if np.any(np.isnan(parameters)):
        multip = np.zeros(6) * np.nan
    else:
        v_0, s_0, T, alpha, beta = parameters.values
        delta = 4.
        # l_leader = cfdata['l_leader'].iloc[0]
        # l_follower = cfdata['l_follower'].iloc[0]
        
        s_star = np.zeros_like(time) * np.nan
        acc_hat = np.zeros_like(time) * np.nan
        spacing_hat = np.zeros_like(time) * np.nan
        speed_hat = np.zeros_like(speed) * np.nan
        speed_hat[0] = speed[0]
        position_hat = np.zeros_like(position) * np.nan
        position_hat[0] = position[0]
        for t in np.arange(0,len(speed_hat)-1,1): # operational time interval is 0.3 second
            s_star[t] = s_0 + max(0., speed_hat[t]*T + speed_hat[t]*(speed_hat[t]-cfdata['v_leader'].iloc[t])/2/np.sqrt(alpha*beta))
            spacing_hat[t] = cfdata['x_leader'].iloc[t] - position_hat[t]# + l_leader/2 - l_follower/2
            if speed_hat[t]<=0. and spacing_hat[t]<s_0:
                acc_hat[t] = 0.
            else:
                acc_hat[t] = alpha * (1 - (speed_hat[t]/v_0)**delta - (s_star[t]/spacing_hat[t])**2)
            speed_hat[t+1] = speed_hat[t] + acc_hat[t] * (time[t+1]-time[t])
            speed_hat[speed_hat<0.] = 0.
            position_hat[t+1] = position_hat[t] + (speed_hat[t]+speed_hat[t+1])/2 * (time[t+1]-time[t])

        speed_hat[(speed<0.1)&(speed_hat<0.1)] = np.nan
        acc_hat[(abs(acceleration)<0.1)&(abs(acc_hat)<0.01)] = np.nan
        multip = [np.nanmean(abs(acceleration[:-1] - acc_hat[:-1])),
                  np.nanmean(abs(speed[1:] - speed_hat[1:])),
                  abs(position[1:] - position_hat[1:]).mean(),
                  np.sqrt((np.nanmean((acceleration[:-1] - acc_hat[:-1])**2))),
                  np.sqrt((np.nanmean((speed[1:] - speed_hat[1:])**2))), 
                  np.sqrt(((position[1:] - position_hat[1:])**2).mean())]
    
    return multip
    

parent_dir = os.path.abspath('..') # Set your parent directory here. 
                                   # Without change the current setting is the parent directory of this file.
data_path = parent_dir + 'Data/OutputData/Variability/'

for cfpair in ['HA','HH']:
    data = pd.read_hdf(data_path+'cfdata_idm_Lyft_'+cfpair+'.h5', key='data')
    data = data.sort_values(['case_id','time']).set_index('case_id')
    metrics = ['MAE_a','MAE_v','MAE_s','RMSE_a','RMSE_v','RMSE_s']
    idm = pd.read_csv(data_path+'idm/parameters_idm_Lyft_'+cfpair+'.csv', index_col=0)
    loss = pd.DataFrame(index=idm.index, columns=metrics)
    for case_id in tqdm(idm.index.values):
        loss.loc[case_id, metrics] = idm_loss(data.loc[case_id],idm.loc[case_id])
    loss.to_csv(data_path+'idm/loss_idm_Lyft_'+cfpair+'.csv')
