'''
Calibrate Gipps' model for selected CF data (applied to Lyft data only).

'''

import os
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def gipps_global_obj(parameters, cfdata):
    v_0, s_0, tau, alpha, b, b_leader = parameters
    id_tau = int(tau/0.1)
    # v_0: free-flow speed
    # s_0: minimum spacing (effective length)
    # tau: reaction time (effective time headway)
    # alpha: max acceleration
    # b: max deceleration of the following vehicle
    # b_leader: max deceleration of the leading vehicle

    position, speed, time = cfdata[['x_follower','v_follower','time']].values.T

    spacing_hat = np.zeros_like(time) * np.nan
    speed_hat = np.zeros_like(speed) * np.nan
    speed_hat[:id_tau] = speed[:id_tau]
    position_hat = np.zeros_like(position) * np.nan
    position_hat[:id_tau] = position[:id_tau]
    for t in np.arange(0,len(speed_hat)-id_tau,1):
        spacing_hat[t] = cfdata['x_leader'].iloc[t] - position_hat[t]
        v_acc = speed_hat[t] + 2.5*alpha*tau*(1-speed_hat[t]/v_0) * np.sqrt(0.025+speed_hat[t]/v_0)
        braking_spacing = 2*(spacing_hat[t]-s_0)-tau*speed_hat[t]+(cfdata['v_leader'].iloc[t])**2/b_leader
        v_dec = -tau*b + np.sqrt(tau**2*b**2 + b*max(0., braking_spacing))
        speed_hat[t+id_tau] = max(0., min(v_acc, v_dec))
        position_hat[t+id_tau] = position_hat[t] + (speed_hat[t]+speed_hat[t+id_tau])/2 * (time[t+id_tau]-time[t])
        
    condition = (speed[id_tau:]>0.001) # make sure the loss computation is not affected by zero speed
    speed = speed[id_tau:][condition]
    speed_hat = speed_hat[id_tau:][condition]
    position = position[id_tau:][condition]
    position_hat = position_hat[id_tau:][condition]

    loss_v = np.sum((speed_hat - speed)**2/abs(speed))/np.sum(abs(speed))
    loss_x = np.sum((position_hat - position)**2/abs(position))/np.sum(abs(position))

    return loss_v + loss_x


def gipps_loss(cfdata, parameters):
    position, speed, time = cfdata[['x_follower','v_follower','time']].values.T

    if np.any(np.isnan(parameters)):
        mae_v = np.nan
    else:
        v_0, s_0, tau, alpha, b, b_leader = parameters
        id_tau = int(tau/0.1)
        
        spacing_hat = np.zeros_like(time) * np.nan
        speed_hat = np.zeros_like(speed) * np.nan
        speed_hat[:id_tau] = speed[:id_tau]
        position_hat = np.zeros_like(position) * np.nan
        position_hat[:id_tau] = position[:id_tau]
        for t in np.arange(0,len(speed_hat)-id_tau,1):
            spacing_hat[t] = cfdata['x_leader'].iloc[t] - position_hat[t]
            v_acc = speed_hat[t] + 2.5*alpha*tau*(1-speed_hat[t]/v_0) * np.sqrt(0.025+speed_hat[t]/v_0)
            braking_spacing = 2*(spacing_hat[t]-s_0)-tau*speed_hat[t]+(cfdata['v_leader'].iloc[t])**2/b_leader
            v_dec = -tau*b + np.sqrt(tau**2*b**2 + b*max(0., braking_spacing))
            speed_hat[t+id_tau] = max(0., min(v_acc, v_dec))
            position_hat[t+id_tau] = position_hat[t] + (speed_hat[t]+speed_hat[t+id_tau])/2 * (time[t+id_tau]-time[t])
            
        mae_v = (abs(speed[id_tau:] - speed_hat[id_tau:])).mean()
    return mae_v


def calibrate_gipps_global(cfdata, v_follower_max, dhw_min, thw_min, thw_median, a_follower):
    
    a_max = max(0.5,min(4.5,np.percentile(a_follower[a_follower>0], 90) if np.any(a_follower>0) else 0.5))
    d_max = max(0.5,min(4.5, np.percentile(-a_follower[a_follower<0], 75) if np.any(a_follower<0) else 0.5))
    res = differential_evolution(gipps_global_obj,
                                 args=(cfdata,),
                                 x0=[v_follower_max,
                                     dhw_min,
                                     thw_median,
                                     a_max,
                                     d_max,
                                     d_max],
                                 bounds=[(12.,min(29., v_follower_max+10.)),
                                         (dhw_min-0.2, 20.),
                                         (max(0.5,thw_min-0.2),10.),
                                         (0.3,a_max+1.5),
                                         (d_max-0.2,6.),
                                         (d_max-0.2,6.)],
                                 popsize=15,
                                 maxiter=750,
                                 workers=15)
    if res.success:
        return res.x
    else:
        return np.zeros(6) * np.nan
    

# parent_dir = './' # Set your parent directory here. 
#                   # Without change the current setting is the parent directory of this file.
# data_path = parent_dir + 'Data/OutputData/Variability/'
data_path = '../OutputData/'

for cfpair in ['HH','HA']:
    data = pd.read_hdf(data_path+'cfdata_idm_Lyft_'+cfpair+'.h5', key='data')
    data = data.sort_values(['case_id','time']).set_index('case_id')

    data['dhw'] = data['x_leader'] - data['x_follower']
    data['thw'] = data['dhw']/data['v_follower']

    case_ids = data.index.unique().values
    savefile = np.zeros(case_ids.shape).astype(bool)
    savefile[np.arange(50,len(case_ids),50)] = True
    savefile[-1] = True

    results = np.zeros((len(case_ids),6))
    i = 0
    progress_bar = tqdm(zip(case_ids[300:], savefile[300:]), total=len(case_ids))
    for case_id,savef in progress_bar:
        cfdata = data.loc[case_id]

        # make the initial position of follower larger than 0
        cfdata['x_leader'] = cfdata['x_leader'] - cfdata['x_follower'].min() + 1.
        cfdata['x_follower'] = cfdata['x_follower'] - cfdata['x_follower'].min() + 1.

        result = calibrate_gipps_global(cfdata, 
                                        max(12.5,cfdata['v_follower'].max()),
                                        min(15., max(4.,cfdata['dhw'].min())),
                                        cfdata['thw'].min(),
                                        cfdata[cfdata['thw']<=10]['thw'].median(), 
                                        cfdata['a_follower'])
        results[i] = result
        i += 1
        if savef:
            results_tosave = pd.DataFrame(results, columns=['v_0','s_0','tau','alpha','b','b_leader'], index=case_ids)
            results_tosave.to_csv(data_path+'gipps/parameters_Lyft_'+cfpair+'.csv')
        
        # progress_bar.set_postfix(loss=gipps_loss(cfdata,result), refresh=False)
