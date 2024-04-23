'''
Standardize the format of both Waymo and Lyft CF datasets.

'''

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm


data_path = './Data/InputData/'


# Waymo CF data

cfdata = pd.read_csv(data_path+'Waymo/all_seg_paired_cf_trj_final_with_large_vehicle.csv')
cfdata = cfdata.rename(columns={'filter_pos':'x','filter_speed':'v','filter_accer':'a'})
cfdata['uniq_id'] = cfdata['segment_id'].astype(int).astype(str) + '-' + cfdata['leader_id'].astype(int).astype(str) + '-' + cfdata['follower_id'].astype(int).astype(str) + '-' + cfdata['local_time'].astype(str)+ '-' + cfdata['local_veh_id'].astype(int).astype(str)

newdata = cfdata[['segment_id','local_time','leader_id','follower_id']].drop_duplicates()
newdata['uniq_leader'] = newdata['segment_id'].astype(int).astype(str) + '-' + newdata['leader_id'].astype(int).astype(str) + '-' + newdata['follower_id'].astype(int).astype(str) + '-' + newdata['local_time'].astype(str) + '-' + newdata['leader_id'].astype(int).astype(str)
newdata['uniq_follower'] = newdata['segment_id'].astype(int).astype(str) + '-' + newdata['leader_id'].astype(int).astype(str) + '-' + newdata['follower_id'].astype(int).astype(str) + '-' + newdata['local_time'].astype(str) + '-' + newdata['follower_id'].astype(int).astype(str)
newdata[['x_leader','v_leader','a_leader','l_leader']] = cfdata.set_index('uniq_id').reindex(newdata.set_index('uniq_leader').index)[['x','v','a','length']].values
newdata[['x_follower','v_follower','a_follower','l_follower']] = cfdata.set_index('uniq_id').reindex(newdata.set_index('uniq_follower').index)[['x','v','a','length']].values
newdata.loc[newdata['leader_id']==0, 'label'] = 'HA'
newdata.loc[newdata['follower_id']==0, 'label'] = 'AH'
newdata.loc[(newdata['leader_id']!=0) & (newdata['follower_id']!=0), 'label'] = 'HH'
newdata['case_id'] = newdata['label'] + '-' + newdata['segment_id'].astype(int).astype(str) + '-' + newdata['leader_id'].astype(int).astype(str) + '-' + newdata['follower_id'].astype(int).astype(str)
newdata = newdata.drop(columns=['uniq_leader','uniq_follower','segment_id','leader_id','follower_id'])
newdata = newdata.rename(columns={'local_time':'time'})

before = newdata.loc[(newdata['case_id']=='HH-269-2-14')&(newdata['time']==17.2), ['x_leader','v_leader','a_leader','l_leader']].values
after = newdata.loc[(newdata['case_id']=='HH-269-2-14')&(newdata['time']==17.4), ['x_leader','v_leader','a_leader','l_leader']].values
newdata.loc[(newdata['case_id']=='HH-269-2-14')&(newdata['time']==17.3), ['x_leader','v_leader','a_leader','l_leader']] = (before+after)/2

newdata[newdata['label']=='AH'].drop(columns=['label']).to_hdf(data_path+'Waymo/CFdata/AH.h5', key='data', mode='w')
newdata[newdata['label']=='HA'].drop(columns=['label']).to_hdf(data_path+'Waymo/CFdata/HA.h5', key='data', mode='w')
newdata[newdata['label']=='HH'].drop(columns=['label']).to_hdf(data_path+'Waymo/CFdata/HH.h5', key='data', mode='w')


# Lyft CF data

def get_data(cfpair, data_path, dataset='train'):

    data = zarr.open(data_path + dataset + cfpair + '.zarr/', mode='r')
    indexrange = data.index_range[:]

    if cfpair == 'AH':
        leadsize = data.lead_size[:]
        followsize = 4.87*np.ones(len(leadsize)) # 4.87 m is given as ground truth
    elif cfpair == 'HA':
        followsize = data.follow_size[:]
        leadsize = 4.87*np.ones(len(followsize))
    else:
        leadsize = data.lead_size[:]
        followsize = data.follow_size[:]

    case_ids = np.zeros(len(data.timestamp)).astype(int)
    leader_size = np.zeros(len(data.timestamp))
    follower_size = np.zeros(len(data.timestamp))
    for case_id in np.arange(len(indexrange)):
        start, end = indexrange[case_id]
        case_ids[start:end] = case_id
        leader_size[start:end] = leadsize[case_id]
        follower_size[start:end] = followsize[case_id]

    data = pd.DataFrame({'case_id':case_ids.astype(int),
                         'time':np.round(data.timestamp,1),
                         'x_leader':data.lead_centroid,
                         'x_follower':data.follow_centroid,
                         'v_leader':data.lead_velocity,
                         'v_follower':data.follow_velocity,
                         'a_leader':data.lead_acceleration,
                         'a_follower':data.follow_acceleration,
                         'l_leader':leader_size,
                         'l_follower':follower_size})
    
    group = data.groupby('case_id').time
    incomplete_case_ids = group.first().index[(10*(group.last()-group.first()+0.1)).astype(int)>group.count()].values

    data = data.set_index('case_id')
    complete_cases = []
    for case_id in tqdm(incomplete_case_ids):
        incomplete_case = data.loc[case_id]
        time = np.round(np.arange(0, incomplete_case.time.max()+0.1, 0.1),1)
        complete_case = pd.DataFrame({'case_id':case_id*np.ones(len(time)).astype(int),
                                      'time':time,
                                      'x_leader':np.interp(time, incomplete_case.time, incomplete_case.x_leader),
                                      'x_follower':np.interp(time, incomplete_case.time, incomplete_case.x_follower),
                                      'v_leader':np.interp(time, incomplete_case.time, incomplete_case.v_leader),
                                      'v_follower':np.interp(time, incomplete_case.time, incomplete_case.v_follower),
                                      'a_leader':np.interp(time, incomplete_case.time, incomplete_case.a_leader),
                                      'a_follower':np.interp(time, incomplete_case.time, incomplete_case.a_follower),
                                      'l_leader':incomplete_case.l_leader.iloc[0]*np.ones(len(time)),
                                      'l_follower':incomplete_case.l_follower.iloc[0]*np.ones(len(time))})
        complete_cases.append(complete_case)
    data = data.reset_index()
    data = pd.concat([data[~data['case_id'].isin(incomplete_case_ids)], pd.concat(complete_cases)], axis=0)
        
    if cfpair == 'HA':
        driver_ids = np.load(data_path+'driver_id_'+dataset+'_HA.npz', allow_pickle = True)
        driver_ids = pd.DataFrame({'case_id':np.arange(len(indexrange)), 'driver_id':driver_ids['ids']}).set_index('case_id')
        data['driver_id'] = driver_ids.loc[data['case_id'].values].values

    return data


for cfpair in ['AH','HA','HH']:
    for dataset in ['train','val']:
        data = get_data(cfpair, data_path+'Lyft5/CFdata/', dataset=dataset)
        group = data.groupby('case_id').time
        print('Number of incomplete trajectories', group.first().index[(10*(group.last()-group.first()+0.1)).astype(int)>group.count()].values.shape)
        data.to_hdf(data_path + 'Lyft5/CFdata/'+cfpair+'_'+dataset+'.h5', key='data', mode='w')

    data_train = pd.read_hdf(data_path + 'Lyft5/CFdata/'+cfpair+'_train.h5', key='data')
    data_val = pd.read_hdf(data_path + 'Lyft5/CFdata/'+cfpair+'_val.h5', key='data')
    data_train['case_id'] = (data_train['case_id'] + 1e5 ).astype(int)
    data_val['case_id'] = (data_val['case_id'] + 3e5 ).astype(int)
    data = pd.concat([data_train, data_val], axis=0).reset_index(drop=True)
    data.to_hdf(data_path + 'Lyft5/CFdata/'+cfpair+'.h5', key='data', mode='w')
