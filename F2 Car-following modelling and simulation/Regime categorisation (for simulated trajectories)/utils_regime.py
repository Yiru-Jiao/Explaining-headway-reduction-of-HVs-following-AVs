'''
Functions for CF regime identification

'''

import numpy as np
from fastdtw import fastdtw

def speed_segmentation(t, v, slope_threshold=1e-3):
    max_num_segments = int((t[-1]-t[0])/0.5)
    num_segments = len(v)
    seg_bool = np.ones(num_segments).astype(bool)
    indices = np.arange(0,len(v))
    t_new = t[seg_bool]
    v_new = v[seg_bool]
    idx_list = indices[seg_bool]
    costs = np.ones(len(v))*999
    while num_segments > max_num_segments:
        # compute the cost of merging adjacent segments
        for i in range(1,len(idx_list)-1):
            adjacent_ts = np.array([t[idx_list[i-1]],
                                    t[idx_list[i]],
                                    t[idx_list[i+1]]])
            
            adjacent_vs = np.array([v[idx_list[i-1]],
                                    v[idx_list[i]],
                                    v[idx_list[i+1]]])
            
            diff = adjacent_vs - np.interp(adjacent_ts, [t[idx_list[i-1]],t[idx_list[i+1]]], [v[idx_list[i-1]],v[idx_list[i+1]]])
            # costs[idx_list[i]] = np.sqrt((diff**2).sum())
            costs[idx_list[i]] = abs(diff).sum()
        
        # identify the pair with the minimum cost and merge
        seg_bool[np.argmin(costs)] = False
        t_new = t[seg_bool]
        v_new = v[seg_bool]
        idx_list = indices[seg_bool]
        num_segments = seg_bool.sum()
        costs = np.ones(len(v))*999

        # further merge the successive segments if they are of the same slope
        slopes = (v_new[1:] - v_new[:-1]) / (t_new[1:] - t_new[:-1])
        diff = slopes[1:] - slopes[:-1]
        idx_to_delete = np.where(abs(diff)<slope_threshold)[0] + 1
        seg_bool[idx_list[idx_to_delete]] = False
        t_new = t[seg_bool]
        v_new = v[seg_bool]
        idx_list = indices[seg_bool]
        num_segments = seg_bool.sum()
        costs = np.ones(len(v))*999
        
    return seg_bool


def cf_or_ff(t, x_leader, x_follower, v_leader, v_follower, miu_limit, sigma_limit):
    # apply DTW and calculate tau based on positions
    _, path = fastdtw(x_leader, x_follower, dist=2)
    tau = np.array([t[p[1]]-t[p[0]] for p in path])
    if len(tau[tau>=0.1])<(len(tau)/2):
        # apply DTW and calculate tau based on speed profiles
        _, path = fastdtw(v_leader, v_follower, dist=2)
        tau = np.array([t[p[1]]-t[p[0]] for p in path])

    # fill in outlier tau (<0.1) with the closest non-outlier value
    try:
        tau[tau<0.1] = [tau[np.where(tau>=0.1)[0][abs(np.where(tau>=0.1)[0]-id).argmin()]] for id in np.where(tau<0.1)[0]]
    except:
        tau[tau<0] = [tau[np.where(tau>=0)[0][abs(np.where(tau>=0)[0]-id).argmin()]] for id in np.where(tau<0)[0]]

    # compress tau list to the same length as t
    tau_new = np.zeros(len(t))
    tau_count = np.zeros(len(t))
    for p in path:
        tau_new[p[1]] = tau_new[p[1]] + tau[p[1]]
        tau_count[p[1]] = tau_count[p[1]] + 1
    tau = tau_new/tau_count

    # obtain mean and std of tau
    tau_prime = np.gradient(tau)
    seg_points = np.where((tau_prime[:-1] * tau_prime[1:])<=0)[0]
    try:
        start, end = seg_points[[0,-1]]
        miu_tau, sigma_tau = tau[start:end+1].mean(), tau[start:end+1].std()
    except:
        miu_tau, sigma_tau = tau.mean(), tau.std()
    # print([miu_tau, sigma_tau])

    # determine threshold
    if (miu_tau > miu_limit) or (sigma_tau > sigma_limit):
        threshold = miu_tau
    else:
        threshold = miu_tau + 2*sigma_tau

    # distinguish cf or ff
    cf = tau<threshold

    # merge isolated points
    cf_shift_left = np.insert(cf[:-1], 0, cf[0])
    cf_shift_right = np.insert(cf[1:], -1, cf[-1])
    while sum((cf!=cf_shift_left)&(cf!=cf_shift_right))!=0:
        cf[(cf!=cf_shift_left)&(cf!=cf_shift_right)] = cf_shift_left[(cf!=cf_shift_left)&(cf!=cf_shift_right)]
        cf_shift_left = np.insert(cf[:-1], 0, cf[0])
        cf_shift_right = np.insert(cf[1:], -1, cf[-1])
    
    ff = np.logical_not(cf)
    
    return cf, ff


def time_regime(t, x_leader, x_follower, v_leader, v_follower, miu_limit, sigma_limit):
    indices = np.arange(0,len(v_follower))
    seg_point = speed_segmentation(t, v_follower)
    cf, ff = cf_or_ff(t, x_leader, x_follower, v_leader, v_follower, miu_limit, sigma_limit)
    regimes = []
    for start, end in zip(indices[seg_point][:-1],indices[seg_point][1:]):
        v_segment = v_follower[start:end+1]
        slope = (v_follower[end] - v_follower[start]) / (t[end] - t[start])

        # FF
        if sum(ff[start:end+1])>sum(cf[start:end+1]):
            if abs(slope)<=0.5 :
                if np.mean(v_segment<0.1):
                    regime = 'S'
                else:
                    regime = 'C'
            elif slope>0.5:
                regime = 'Fa'
            elif slope<-0.5:
                regime = 'Fd'
        else: # CF
            if abs(slope)<=0.5 :
                if np.mean(v_segment<0.1):
                    regime = 'S'
                else:
                    regime = 'F'
            elif slope>0.5:
                regime = 'A'
            elif slope<-0.5:
                regime = 'D'
        regimes.extend([regime]*(end-start))
    regimes.append(regimes[-1])
        
    return regimes
