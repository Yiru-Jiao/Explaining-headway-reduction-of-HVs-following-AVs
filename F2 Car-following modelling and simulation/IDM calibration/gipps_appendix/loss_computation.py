'''
Compute calibration loss of Gipps' model

'''

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def gipps_loss(cfdata,parameters):
    position, speed, time = cfdata[['x_follower','v_follower','time']].values.T

    if np.any(np.isnan(parameters)):
        multip = np.zeros(6) * np.nan
    else:
        v_0, s_0, tau, alpha, b, b_leader = parameters
        theta = tau/2
        id_tau = int(tau/0.1)
        
        spacing_hat = np.zeros_like(time) * np.nan
        speed_hat = np.zeros_like(speed) * np.nan
        speed_hat[:id_tau] = speed[:id_tau]
        position_hat = np.zeros_like(position) * np.nan
        position_hat[:id_tau] = position[:id_tau]
        for t in np.arange(0,len(speed_hat)-id_tau,1): # update every 0.1 second
            spacing_hat[t] = cfdata['x_leader'].iloc[t] - position_hat[t]
            v_acc = speed[t] + 2.5*alpha*tau*(1-speed[t]/v_0) * np.sqrt(0.025+speed[t]/v_0)
            v_dec = -(tau+theta)*b + np.sqrt((tau+theta)**2*b**2 + b*(2*(spacing_hat[t]-s_0)-tau*speed[t]+(cfdata['v_leader'].iloc[t])**2/b_leader))
            speed_hat[t+id_tau] = max(0., min(v_acc, v_dec))
            position_hat[t+id_tau] = position_hat[t+id_tau-1] + (speed_hat[t+id_tau-1]+speed_hat[t+id_tau])/2 * (time[t+id_tau]-time[t+id_tau-1])

        speed_hat[(speed<0.01)&(speed_hat<0.01)] = np.nan # it's not meaningful to compute loss when speed is near zero
        multip = [np.nanmean(abs(speed[id_tau:] - speed_hat[id_tau:])),
                  abs(position[id_tau:] - position_hat[id_tau:]).mean(),
                  np.sqrt((np.nanmean((speed[id_tau:] - speed_hat[id_tau:])**2))), 
                  np.sqrt(((position[id_tau:] - position_hat[id_tau:])**2).mean())]
    
    return multip
    

parent_dir = './' # Set your parent directory here. 
                  # Without change the current setting is the parent directory of this file.
data_path = parent_dir + 'Data/OutputData/Variability/'

for cfpair in ['HA','HH']:
    data = pd.read_hdf(data_path+'cfdata_idm_Lyft_'+cfpair+'.h5', key='data')
    data = data.sort_values(['case_id','time']).set_index('case_id')
    metrics = ['MAE_v','MAE_s','RMSE_v','RMSE_s']
    gipps = pd.read_csv(data_path+'gipps/parameters_gipps_Lyft_'+cfpair+'.csv', index_col=0)
    loss = pd.DataFrame(index=gipps.index, columns=metrics)
    for case_id in tqdm(gipps.index.values):
        loss.loc[case_id, metrics] = gipps_loss(data.loc[case_id],gipps.loc[case_id])
    loss.to_csv(data_path+'gipps/loss_gipps_Lyft_'+cfpair+'.csv')
