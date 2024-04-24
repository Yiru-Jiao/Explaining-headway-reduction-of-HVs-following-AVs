'''
Select CF data with complete regimes.

'''

import os
import numpy as np
import pandas as pd

parent_dir = os.path.abspath('..') # Set your parent directory here. 
                                   # Without change the current setting is the parent directory of this file.


def get_data(cfpair, dataset):
    data_path = parent_dir + 'Data/InputData/'
    regime_path = parent_dir + 'Data/OutputData/CF regime/'

    if dataset=='Waymo':
        cfdata = pd.read_hdf(data_path+'Waymo/CFdata/'+cfpair+'.h5', key='data')
        regimes = pd.read_csv(regime_path+'Waymo/regimes/regimes_list_'+cfpair+'.csv')
    elif dataset=='Lyft':
        cfdata = pd.read_hdf(data_path+'Lyft5/CFdata/'+cfpair+'.h5', key='data')
        regimes = pd.read_csv(regime_path+'Lyft/regimes/regimes_list_'+cfpair+'.csv')

    # remove slightly negative speeds due to filtering
    cfdata.loc[(cfdata['v_follower']<0),'v_follower'] = 0. # negative follower speed can be set to zero because calibration only uses initial states
    cfdata.loc[(cfdata['v_leader']<0)&(cfdata['v_leader']>-0.1),'v_leader'] = 0. # negative leader speed needs to be further filtered

    # select cases include not only free-following
    cfdata = cfdata[cfdata.groupby('case_id')['v_follower'].transform('min')<0.1]

    # complete regimes for calibrating IDM: {Fa, A, D, F} or {S, A, D, F}
    condition = np.all(regimes[['Fa','A','D','F']]>0.5,axis=1) | np.all(regimes[['S','A','D','F']]>0.5,axis=1)
    cfdata = cfdata[cfdata['case_id'].isin(regimes[condition]['case_id'].values)]

    return cfdata
    

data_path = parent_dir + 'Data/OutputData/Variability/'


# select data
for dataset in ['Waymo', 'Lyft']:
    for cfpair in ['HA','HH']:
        data = get_data(cfpair, dataset)

        # assumption 1: leading speed should be non-negative all the time
        data = data.loc[~data['case_id'].isin(data[data.v_leader<0].index.unique())]
        # assumption 2: leading speed cannot be all zero
        non_negative = data.groupby('case_id')['v_leader'].max()
        data = data.loc[data['case_id'].isin(non_negative[non_negative>0.1].index)]
        # assumption 3: initial state condition (will all be fulfilled after removing overly small gaps)
        data['gap'] = data['x_leader'] - data['x_follower'] - data['l_leader']/2 - data['l_follower']/2
        data = data.loc[~data['case_id'].isin(data[data['gap']<0]['case_id'].unique())]
        # calibration preferences 1) max speed of the follower should not exceed speed limit (29 m/s)
        #                         2) max speed of the follower should not be too small (<10 m/s)
        #                         3) max acceleration of the follower should not be too large (>9.8 m/s^2, fulfilled by data preprocessing)
        #                         4) max deceleration of the follower should not be too large (<-6.5 m/s^2, fulfilled by data preprocessing)
        #                         5) the same limits of largest speed and acceleration apply to the leader
        data = data.loc[~data['case_id'].isin(data[data['v_follower']>29.]['case_id'].unique())]
        data = data.loc[data['case_id'].isin(data[data['v_follower']>=10.]['case_id'].unique())]
        data = data.loc[~data['case_id'].isin(data[data['v_leader']>29.]['case_id'].unique())]

        print(dataset, cfpair, len(data['case_id'].unique()), '{:.2f}'.format(((data.l_leader + data.l_follower)/2).mean()))
        data.to_hdf(data_path+'cfdata_idm_'+dataset+'_'+cfpair+'.h5', key='data')

# avoid overlong cases in HA (applied to Lyft only)
data_HA = pd.read_hdf(data_path+'cfdata_idm_Lyft_HA.h5', key='data')
data_HH = pd.read_hdf(data_path+'cfdata_idm_Lyft_HH.h5', key='data')

duration_HA = data_HA.groupby('case_id')['time'].count()
duration_HH = data_HH.groupby('case_id')['time'].count()
print('Longest duration', duration_HH.max())
duration_HA = duration_HA[duration_HA<=duration_HH.max()]
data_HA = data_HA[data_HA['case_id'].isin(duration_HA.index)]
data_HA.to_hdf(data_path+'cfdata_idm_Lyft_HA.h5', key='data')

print('HH', len(data_HH['case_id'].unique()), '{:.2f}'.format(((data_HH.l_leader + data_HH.l_follower)/2).mean()))
print('HA', len(data_HA['case_id'].unique()), '{:.2f}'.format(((data_HA.l_leader + data_HA.l_follower)/2).mean()))
