'''
Calibrate IDM for selected CF data (applied to Lyft data only).

'''

import os
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def idm_global_obj(parameters, cfdata):
    # v_0, s_0, delta, T, alpha, beta = parameters
    v_0, s_0, T, alpha, beta = parameters
    delta = 4.
    # v_0: free-flow speed
    # s_0: minimum spacing (distance headway)
    # delta: acceleration exponent
    # T: safe time headway
    # alpha: max acceleration
    # beta: comfortable deceleration

    position, speed, time = cfdata[['x_follower','v_follower','time']].values.T
    # l_leader = cfdata['l_leader'].iloc[0]
    # l_follower = cfdata['l_follower'].iloc[0]

    s_star = np.zeros_like(time) * np.nan
    acc_hat = np.zeros_like(time) * np.nan
    spacing_hat = np.zeros_like(time) * np.nan
    speed_hat = np.zeros_like(speed) * np.nan
    speed_hat[0] = speed[0]
    position_hat = np.zeros_like(position) * np.nan
    position_hat[0] = position[0]
    for t in np.arange(0,len(speed_hat),3): # operational time interval is 0.3 second
        s_star[t] = s_0 + max(0., speed_hat[t]*T + speed_hat[t]*(speed_hat[t]-cfdata['v_leader'].iloc[t])/2/np.sqrt(alpha*beta))
        spacing_hat[t] = cfdata['x_leader'].iloc[t] - position_hat[t]# + l_leader/2 - l_follower/2
        if speed_hat[t]<=0. and spacing_hat[t]<s_0:
            acc_hat[t] = 0.
        else:
            acc_hat[t] = alpha * (1 - (speed_hat[t]/v_0)**delta - (s_star[t]/spacing_hat[t])**2)
        speed_hat[t+1:t+4] = speed_hat[t] + acc_hat[t] * (time[t+1:t+4]-time[t])
        speed_hat[speed_hat<0.] = 0.
        position_hat[t+1:t+4] = position_hat[t] + (speed_hat[t]+speed_hat[t+1:t+4])/2 * (time[t+1:t+4]-time[t])
        
    speed[speed==0.] = np.nan
    loss_v = np.nansum((speed_hat[1:] - speed[1:])**2/abs(speed[1:]))/np.nansum(abs(speed[1:]))
    loss_x = np.nansum((position_hat[1:] - position[1:])**2/abs(position[1:]))/np.nansum(abs(position[1:]))

    return loss_v + loss_x


def idm_loss(cfdata,parameters):
    position, speed, time = cfdata[['x_follower','v_follower','time']].values.T

    if np.any(np.isnan(parameters)):
        mae_v = np.nan
    else:
        v_0, s_0, T, alpha, beta = parameters
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
        for t in np.arange(0,len(speed_hat),3): # operational time interval is 0.3 second
            s_star[t] = s_0 + max(0., speed_hat[t]*T + speed_hat[t]*(speed_hat[t]-cfdata['v_leader'].iloc[t])/2/np.sqrt(alpha*beta))
            spacing_hat[t] = cfdata['x_leader'].iloc[t] - position_hat[t]# + l_leader/2 - l_follower/2
            if speed_hat[t]<=0. and spacing_hat[t]<s_0:
                acc_hat[t:t+3] = 0.
            else:
                acc_hat[t:t+3] = alpha * (1 - (speed_hat[t]/v_0)**delta - (s_star[t]/spacing_hat[t])**2)
            speed_hat[t+1:t+4] = speed_hat[t] + acc_hat[t] * (time[t+1:t+4]-time[t])
            speed_hat[speed_hat<0.] = 0.
            position_hat[t+1:t+4] = position_hat[t] + (speed_hat[t]+speed_hat[t+1:t+4])/2 * (time[t+1:t+4]-time[t])

            mae_v = abs(speed[1:] - speed_hat[1:]).mean()
    return mae_v


def calibrate_idm_global(cfdata, v_follower_max, dhw_min, thw_min, thw_median, a_follower):
    
    a_max = max(0.5,min(4.5,np.percentile(a_follower[a_follower>0], 90) if np.any(a_follower>0) else 0.5))
    d_max = max(0.5,min(4.5, np.percentile(-a_follower[a_follower<0], 75) if np.any(a_follower<0) else 0.5))
    # print(a_max, d_max, v_follower_max, dhw_min, thw_min, thw_median)
    res = differential_evolution(idm_global_obj,
                                 args=(cfdata,),
                                 x0=[v_follower_max,
                                     dhw_min,
                                     thw_median,
                                     a_max,
                                     d_max],
                                 bounds=[(12.,min(29., v_follower_max+10.)),
                                         (dhw_min-0.2, 20.),
                                         (max(0.5,thw_min-0.2),10.),
                                         (0.3,a_max+1.5),
                                         (d_max-0.2,6.)],
                                #  bounds=[(10.,29.),
                                #          (3.5, 20.),
                                #          (0.5,10.),
                                #          (0.5,6.),
                                #          (0.5,6.)],
                                 popsize=15,
                                 maxiter=750,
                                 workers=15)
    
    # res = minimize(idm_global_obj, 
    #                args=(cfdata,),
    #                x0=[v_follower_max,
    #                    dhw_min,
    #                    thw_median,
    #                    a_max,
    #                    d_median],
    #                bounds=[(10.,min(29., v_follower_max+10.)),
    #                        (dhw_min-0.2, 20.),
    #                        (max(0.5,thw_min-0.2),10.),
    #                        (a_max-2,a_max+2),
    #                        (d_median-2,d_median+2)],
    #                method='Nelder-Mead',
    #                tol=1e-6
    #                )

    if res.success:
        return res.x
    else:
        return np.zeros(5) * np.nan
    

data_path = os.path.abspath('../..') + '/OutputData/CFAV/'

for cfpair in ['HH','HA']:
    data = pd.read_hdf(data_path+'cfdata_idm_Lyft_'+cfpair+'.h5', key='data')
    data = data.sort_values(['case_id','time']).set_index('case_id')

    data['dhw'] = data['x_leader'] - data['x_follower']# - data['l_follower']/2 + data['l_leader']/2
    data['thw'] = data['dhw']/data['v_follower']

    case_ids = data.index.unique().values
    savefile = np.zeros(case_ids.shape).astype(bool)
    savefile[np.arange(50,len(case_ids),50)] = True
    savefile[-1] = True

    results = np.zeros((len(case_ids),5))
    i = 0
    progress_bar = tqdm(zip(case_ids, savefile), total=len(case_ids))
    for case_id,savef in progress_bar:
        cfdata = data.loc[case_id]

        # make the initial position of follower larger than 0
        cfdata['x_follower'] = cfdata['x_follower'] - cfdata['x_follower'].min() + 1.
        cfdata['x_leader'] = cfdata['x_leader'] - cfdata['x_follower'].min() + 1.

        result = calibrate_idm_global(cfdata, 
                                      max(12.5,cfdata['v_follower'].max()),
                                      min(15., max(4.,cfdata['dhw'].min())),
                                      cfdata['thw'].min(),
                                      cfdata[cfdata['thw']<=10]['thw'].median(), 
                                      cfdata['a_follower'])
        results[i] = result
        i += 1
        if savef:
            results_tosave = pd.DataFrame(results, columns=['v_0','s_0','T','alpha','beta'], index=case_ids)
            results_tosave.to_csv(data_path+'parameters_idm_Lyft_'+cfpair+'.csv')
        
        progress_bar.set_postfix(loss=idm_loss(cfdata,result), refresh=False)
