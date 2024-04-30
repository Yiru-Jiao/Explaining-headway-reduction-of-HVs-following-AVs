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
        id_tau = int(tau/0.1)
        
        spacing_hat = np.zeros_like(time) * np.nan
        speed_hat = np.zeros_like(speed) * np.nan
        speed_hat[:id_tau] = speed[:id_tau]
        position_hat = np.zeros_like(position) * np.nan
        position_hat[:id_tau] = position[:id_tau]
        for t in np.arange(0,len(speed_hat)-id_tau,1): # update every 0.1 second
            spacing_hat[t] = cfdata['x_leader'].iloc[t] - position_hat[t]
            v_acc = speed_hat[t] + 2.5*alpha*tau*(1-speed_hat[t]/v_0) * np.sqrt(0.025+speed_hat[t]/v_0)
            braking_spacing = 2*(spacing_hat[t]-s_0)-tau*speed_hat[t]+(cfdata['v_leader'].iloc[t])**2/b_leader
            v_dec = -tau*b + np.sqrt(tau**2*b**2 + b*max(0., braking_spacing))
            speed_hat[t+id_tau] = max(0., min(v_acc, v_dec))
            position_hat[t+id_tau] = position_hat[t] + (speed_hat[t]+speed_hat[t+id_tau])/2 * (time[t+id_tau]-time[t])

        condition = (speed[id_tau:]>0.)|(speed_hat[id_tau:]>0.) # exclude comparison when both speed and speed_hat are zero
        speed = speed[id_tau:][condition]
        speed_hat = speed_hat[id_tau:][condition]
        position = position[id_tau:][condition]
        position_hat = position_hat[id_tau:][condition]

        multip = [np.mean(abs(speed - speed_hat)),
                  abs(position - position_hat).mean(),
                  np.sqrt((np.mean((speed - speed_hat)**2))), 
                  np.sqrt((np.mean((position - position_hat)**2)))]
    
    return multip
    

parent_dir = './' # Set your parent directory here. 
                  # Without change the current setting is the parent directory of this file.
data_path = parent_dir + 'Data/OutputData/Variability/'

for cfpair in ['HA','HH']:
    data = pd.read_hdf(data_path+'cfdata_idm_Lyft_'+cfpair+'.h5', key='data')
    data = data.sort_values(['case_id','time']).set_index('case_id')
    metrics = ['MAE_v','MAE_s','RMSE_v','RMSE_s']
    gipps = pd.read_csv(data_path+'gipps/parameters_Lyft_'+cfpair+'.csv', index_col=0)
    loss = pd.DataFrame(index=gipps.index, columns=metrics)
    for case_id in tqdm(gipps.index.values):
        loss.loc[case_id, metrics] = gipps_loss(data.loc[case_id],gipps.loc[case_id])
    loss.to_csv(data_path+'gipps/loss_Lyft_'+cfpair+'.csv')
