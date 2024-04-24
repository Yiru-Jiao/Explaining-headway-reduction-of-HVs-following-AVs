'''
Functions for figure making.

'''

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, gaussian_kde
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



def rgb_color(colorname, alpha=1.):
    color = mpl.colors.to_rgb(colorname)
    return (color[0], color[1], color[2], alpha)



def plot_headway_subplot(ax, data, bins, x_range, color_name, label_prefix, text_pos, linestyle, text=True, method='auto'):
    ax.hist(data, bins=bins, density=True,
            fc=rgb_color(color_name, 0.1), ec=rgb_color(color_name, 0.15), lw=0.5,
            label=label_prefix)
    kernel = gaussian_kde(data, bw_method=method)
    mean = data.mean()
    std = data.std()
    ax.plot(x_range, kernel(x_range), c=color_name, lw=1, ls=linestyle, label=label_prefix)
    ax.plot([mean, mean], [0, ax.get_ylim()[1]], c=color_name, lw=1, ls=':')
    if text:
        ax.text(mean+text_pos, 1.06 * ax.get_ylim()[1], f'{mean:.2f}$\pm${std:.2f}',
                va='center', ha='center', color=color_name)
    


def ks_test(sample_HH, sample_HA, text1, text2):
    _, p_equal = ks_2samp(sample_HH, sample_HA, alternative='two-sided')
    if p_equal>=0.05:
        text = text1 + ' = ' + text2
    else:
        text = text1 + ' ≠ ' + text2
    return p_equal, text



def compare_HH_HA(cfdata_HH, cfdata_HA):
    fig, axes = plt.subplots(1,5,figsize=(7.5,1.1),constrained_layout=True,gridspec_kw={'width_ratios': [1,1,0.05,1,1]})
    axes[2].axis('off')
    
    sample_list = [cfdata_HH.groupby('case_id')['v_leader'].max(),
                   cfdata_HA.groupby('case_id')['v_leader'].max(),
                   cfdata_HH.groupby('case_id')['v_follower'].max(),
                   cfdata_HA.groupby('case_id')['v_follower'].max(),
                   cfdata_HH.groupby('case_id')['a_leader'].agg(['max','min']).values.flatten(),
                   cfdata_HA.groupby('case_id')['a_leader'].agg(['max','min']).values.flatten(),
                   cfdata_HH.groupby('case_id')['a_follower'].agg(['max','min']).values.flatten(),
                   cfdata_HA.groupby('case_id')['a_follower'].agg(['max','min']).values.flatten()]
    ax_list = [axes[0], axes[0],
               axes[1], axes[1],
               axes[3], axes[3],
               axes[4], axes[4]]
    bin_list = [np.linspace(0,24,24), np.linspace(0,24,24),
                np.linspace(0,24,24), np.linspace(0,24,24),
                np.linspace(-5,5,24), np.linspace(-5,5,24),
                np.linspace(-5,5,24), np.linspace(-5,5,24)]
    label_list = ['HH', 'HA',
                  'HH', 'HA',
                  'HH', 'HA',
                  'HH', 'HA']
    fc_list = [rgb_color('tab:blue',0.2), rgb_color('tab:red',0.2),
               rgb_color('tab:blue',0.2), rgb_color('tab:red',0.2),
               rgb_color('tab:blue',0.2), rgb_color('tab:red',0.2),
               rgb_color('tab:blue',0.2), rgb_color('tab:red',0.2)]
    ec_list = [rgb_color('tab:blue',1), rgb_color('tab:red',1),
               rgb_color('tab:blue',1), rgb_color('tab:red',1),
               rgb_color('tab:blue',1), rgb_color('tab:red',1),
               rgb_color('tab:blue',1), rgb_color('tab:red',1)]
    ls_list = ['-.',':',
               '-','--',
               '-.',':',
               '-','--']

    for sample, ax, bins, label, fc, ec, ls in zip(sample_list, ax_list, bin_list, label_list, fc_list, ec_list, ls_list):
        ax.hist(sample, bins=bins, density=True,
                histtype='stepfilled', fc=fc, ec=ec, lw=0.8, ls=ls, label=label)
        ax.legend(loc='upper left', fontsize=8, frameon=False, handlelength=1.2, handletextpad=0.2, bbox_to_anchor=(-0.02, 1.04))

    axes[0].set_ylabel('Probability density')
    for idx,title in zip(range(2),['Leaders','Followers']):
        axes[idx].set_title(title, fontsize=8, pad=0)
        axes[idx+3].set_title(title, fontsize=8, pad=0)

    ax = fig.add_axes([0.06, 0.0, 0.44, 0.96], frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Speed distribution of', fontsize=8)
    ax.set_xlabel('Maximum speed (m/s)', labelpad=1.5)
    
    ax = fig.add_axes([0.56, 0.0, 0.44, 0.96], frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Acc/deceleration distribution of', fontsize=8)
    ax.set_xlabel('Maximum acceleration and deceleration (m/s$^2$)', labelpad=-0.5)

    for ax in [axes[0],axes[1]]:
        ax.set_ylim([0,0.32])
        ax.set_yticks([])
        ax.set_xticks([0,5,10,15,20])
    axes[0].set_yticks([0,0.1,0.2,0.3])
    for ax in [axes[3],axes[4]]:
        ax.set_ylim([0,0.64])
        ax.set_yticks([])
        ax.set_xticks([-5,-2.5,0,2.5,5])
    axes[3].set_yticks([0,0.2,0.4,0.6])  
    return fig, axes



def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0):
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(' ', cdict)

    return newcmap



def density_plot(ax, data, header, xvar, yvar, xbins, ybins, vmax=1., condition=False, cmap=shiftedColorMap(mpl.cm.Reds,midpoint=0.4)):
    p_x, x = np.histogram(data[xvar], bins=xbins, density=True)
    p_y, y = np.histogram(data[yvar], bins=ybins, density=True)
    p_xy, x, y = np.histogram2d(data[xvar], data[yvar], bins=[xbins, ybins], density=True)
    p_xy = p_xy.T
    p_x, p_y = np.meshgrid(p_x, p_y)
    x, y = np.meshgrid(x, y)

    if condition:
        im=ax.pcolormesh(x, y, p_xy/p_x, cmap=cmap, vmin=0, vmax=vmax)
    else:
        im=ax.pcolormesh(x, y, p_xy, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_aspect((xbins.max()-xbins.min())/(ybins.max()-ybins.min()))
    ax.set_title(header, fontsize=8)
    return ax, im



def length_dist(Lyft_HH, Lyft_HA, setname):
    if setname=='Lyft':
        fig, axes = plt.subplots(1,3, figsize=(4,1.6), sharex=True, sharey=True, constrained_layout=True)
    elif setname=='Waymo':
        fig, axes = plt.subplots(1,3, figsize=(3.5,1.6), sharex=True, sharey=True, constrained_layout=True)
        for ax in axes:
            ax.set_xlim([3.5,8.5])
            ax.set_ylim([3,7])
            ax.set_aspect(5/4)

    test = Lyft_HH.copy()
    test['gap'] = test['x_leader'] - test['x_follower'] - test['l_follower']/2 + test['l_leader']/2
    test = test.groupby('case_id').agg({'gap':'min','l_follower':'mean'})
    if setname=='Lyft':
        axes[0], im = density_plot(axes[0], test, 'HV follower in HH', 'gap', 'l_follower', np.linspace(3.5,8.5,25), np.linspace(3,7,25), vmax=0.3)
    elif setname=='Waymo':
        axes[0].scatter(test['gap'], test['l_follower'], s=10, marker='x', lw=0.7, color=mpl.colormaps.get_cmap('Reds')(150), alpha=0.5)
        axes[0].set_title('HV follower in HH', fontsize=8)

    test = Lyft_HA.copy()
    test['gap'] = test['x_leader'] - test['x_follower'] - test['l_follower']/2 + test['l_leader']/2
    test = test.groupby('case_id').agg({'gap':'min','l_follower':'mean'})
    if setname=='Lyft':
        axes[1], im = density_plot(axes[1], test, 'HV follower in HA', 'gap', 'l_follower', np.linspace(3.5,8.5,25), np.linspace(3,7,25), vmax=0.3)
    elif setname=='Waymo':
        axes[1].scatter(test['gap'], test['l_follower'], s=10, marker='x', lw=0.7, color=mpl.colormaps.get_cmap('Reds')(150), alpha=0.5)
        axes[1].set_title('HV follower in HA', fontsize=8)

    test = Lyft_HH.copy()
    test['gap'] = test['x_leader'] - test['x_follower'] - test['l_follower']/2 + test['l_leader']/2
    test = test.groupby('case_id').agg({'gap':'min','l_leader':'mean'})
    if setname=='Lyft':
        axes[2], im = density_plot(axes[2], test, 'HV leader in HH', 'gap', 'l_leader', np.linspace(3.5,8.5,25), np.linspace(3,7,25), vmax=0.3)
    elif setname=='Waymo':
        axes[2].scatter(test['gap'], test['l_leader'], s=10, marker='x', lw=0.7, color=mpl.colormaps.get_cmap('Reds')(150), alpha=0.5)
        axes[2].set_title('HV leader in HH', fontsize=8)

    axes[1].set_xlabel('Minimum gap (m)',labelpad=0)
    axes[0].set_ylabel('Vehicle length (m)')
    for ax in [axes[0],axes[1],axes[2]]:
        ax.set_xticks([4,6,8])
    
    if setname=='Lyft':
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Probability density', shrink=0.64, pad=0.02)

    return fig, axes



def length_impact(cfdata_HH, cfdata_HA):
    fig, axes = plt.subplots(figsize=(2.5,1.45),sharex=True,sharey=True,constrained_layout=True)
    cfdata_HH = cfdata_HH.groupby('case_id')[['l_follower','l_leader']].first()
    cfdata_HA = cfdata_HA.groupby('case_id')[['l_follower','l_leader']].first()

    axes.set_xlabel('$l_{\mathrm{follower}}+l_{\mathrm{leader}}$ (m)',labelpad=0)
    histdata = (cfdata_HH.l_follower + cfdata_HH.l_leader)
    axes.hist(histdata,
              bins=np.arange(5.9, 12.2, 0.25), 
              weights=np.zeros(len(cfdata_HH))+1./len(cfdata_HH),
              fc=rgb_color('tab:blue',0.4), ec=rgb_color('tab:blue',1), lw=0.5, 
              label='HH')
    histdata = (cfdata_HA.l_follower + cfdata_HA.l_leader)
    axes.hist(histdata,
              bins=np.arange(5.9, 12.2, 0.25), 
              weights=np.zeros(len(cfdata_HA))+1./len(cfdata_HA),
              fc=rgb_color('tab:red',0.4), ec=rgb_color('tab:red',1), lw=0.5, ls='--',
              label='HA')
    axes.legend(frameon=False)    
    axes.set_ylabel('Relative frequency')
    axes.set_ylim([0.,0.41])
    axes.set_yticks([0,0.2,0.4])

    return fig, axes



# def correct_av_length(Lyft_HA):
#     test = Lyft_HA[(Lyft_HA['v_leader']<0.1)&(Lyft_HA['v_follower']<0.1)]
#     distance = 2*(test['x_leader'] - test['x_follower'] - test['l_follower']/2)
#     fig, ax = plt.subplots(figsize=(3.5,1.5))
#     _ = ax.hist(distance, bins=np.arange(3.475,6.05,0.05), fc=rgb_color('tab:purple',0.4), ec=rgb_color('tab:purple',0.8), lw=0.5, )
#     ax.plot([4.87,4.87],[0,2900], color='b', ls='-.', lw=0.7, label='Given length')
#     ax.plot([4.5,4.5],[0,2900], color='r', ls='--', lw=0.7, label='Selected length')
#     ax.set_ylim([0,2900])
#     ax.set_xlabel('AV length + 2×gap (m)')
#     ax.set_ylabel('Frequency (moment)')
#     ax.legend(frameon=False, fontsize=8, loc='upper left', handlelength=1.5)
#     return fig, ax



def headway_dist(cfdata_HH, cfdata_HA):
    fig, axes = plt.subplots(1,3,figsize=(5.5,1.3),gridspec_kw={'width_ratios':[1,0.05,1]})
    axes[1].axis('off')
    axes[0].set_xlabel('Min. THW (s)', labelpad=0.5)
    axes[0].set_ylabel('Probability density')
    axes[0].set_ylim([0.,1.25])
    axes[0].set_xticks([1,3,5])
    axes[0].set_yticks([0,0.5,1.])
    axes[2].set_xlabel('Follower speed (m/s)', labelpad=0.5)
    axes[2].set_ylabel('Min. THW (s)')
    axes[2].set_ylim([0.,6.5])
    axes[2].set_yticks([1,3,5])
    axes[2].set_xlim([1,24])
    axes[2].set_xticks([5,10,15,20])

    thw_bins = np.linspace(0, 6, 45)
    thw_range = np.linspace(0, 6, 100)

    thw_HH = cfdata_HH.loc[cfdata_HH.groupby('case_id')['thw'].idxmin()]
    thw_HA = cfdata_HA.loc[cfdata_HA.groupby('case_id')['thw'].idxmin()]

    plot_headway_subplot(axes[0], thw_HH['thw'], thw_bins, thw_range, 'tab:blue', 'HH', 0.95, '-', method=0.2)
    plot_headway_subplot(axes[0], thw_HA['thw'], thw_bins, thw_range, 'tab:red', 'HA', -0.9, '--', method=0.2)
    # pvalue, result = ks_test(thw_HH['thw'], thw_HA['thw'])
    # axes[0].text(0.99, 0.53, f'K-S test p-value: {pvalue:.3f}', transform=axes[0].transAxes, va='center', ha='right')
    # axes[0].text(0.99, 0.4, result, transform=axes[0].transAxes, va='center', ha='right')
    axes[2].plot(thw_HH['v_follower'], thw_HH['thw'], 's', ms=1.5, fillstyle='none', mew=0.2, c='tab:blue', label='HH', rasterized=True)
    axes[2].plot(thw_HA['v_follower'], thw_HA['thw'], 'o', ms=1.5, fillstyle='none', mew=0.2, c='tab:red', label='HA', rasterized=True)
    
    handles, _ = axes[0].get_legend_handles_labels()
    axes[0].legend([(handles[0],handles[1]),(handles[2],handles[3])],
                   ['HH','HA'],
                   frameon=False, loc='upper right')

    axes[2].legend([Line2D([], [], color='none', marker='s', markerfacecolor='none', markeredgecolor=rgb_color('tab:blue'), markersize=3),
                    Line2D([], [], color='none', marker='o', markerfacecolor='none', markeredgecolor=rgb_color('tab:red'), markersize=3)],
                   ['HH','HA'],
                   frameon=False, loc='upper right', handletextpad=0.05, labelspacing=0.1)
    
    return fig, axes



def regime_example(cfdata_HA):
    fig, axes = plt.subplots(1,3,figsize=(6.5,1.4),constrained_layout=True)
    example = cfdata_HA[(cfdata_HA['case_id']==112826)].copy().iloc[20:]
    axes[1].set_title('Example of car-following regimes')
    axes[0].set_ylabel('Position (m)', labelpad=0.5)
    axes[0].set_xlabel('Time (s)', labelpad=0.5)
    axes[1].set_ylabel('Speed (m/s)', labelpad=0.5)
    axes[1].set_xlabel('Time (s)', labelpad=0.5)
    axes[2].set_ylabel('Acceleration (m/s$^2$)', labelpad=0.5)
    axes[2].set_xlabel('Time (s)', labelpad=0.5)
    axes[0].plot(example['time'], example['x_leader'], c=(0,0,0,0.5), lw=1, ls='--', label='Leader')
    axes[1].plot(example['time'], example['v_leader'], c=(0,0,0,0.5), lw=1, ls='--')
    axes[2].plot(example['time'], example['a_leader'], c=(0,0,0,0.5), lw=1, ls='--')
    for regime, cidx, adjustment in zip(['C','A','D','F','S','Fa','Fd'], np.arange(0.,8.)/7, [1,1,0.2,1.5,0,1.8,-0.1]):
        examp = example[(example['regime']==regime)&(example['regime_id']>0)]
        axes[0].plot(examp['time'], examp['x_follower'], c=mpl.colormaps.get_cmap('turbo')(cidx), lw=1.3, label='Follower')
        axes[0].scatter(examp['time'].iloc[0], examp['x_follower'].iloc[0], color=mpl.colormaps.get_cmap('turbo')(cidx), s=15, lw=1, marker='+', label='Start')
        axes[0].text(examp['time'].mean()+adjustment, examp['x_follower'].mean()-5, regime, color=mpl.colormaps.get_cmap('turbo')(cidx), ha='center', va='top')
        axes[1].plot(examp['time'], examp['v_follower'], c=mpl.colormaps.get_cmap('turbo')(cidx), lw=1.3)
        axes[1].scatter(examp['time'].iloc[0], examp['v_follower'].iloc[0], color=mpl.colormaps.get_cmap('turbo')(cidx), s=15, lw=1, marker='+')
        axes[1].text(examp['time'].mean()+adjustment, examp['v_follower'].mean(), regime, color=mpl.colormaps.get_cmap('turbo')(cidx), ha='center', va='top')
        axes[2].plot(examp['time'], examp['a_follower'], c=mpl.colormaps.get_cmap('turbo')(cidx), lw=1.3)
        axes[2].scatter(examp['time'].iloc[0], examp['a_follower'].iloc[0], color=mpl.colormaps.get_cmap('turbo')(cidx), s=15, lw=1, marker='+')
        axes[2].text(examp['time'].mean()+adjustment, examp['a_follower'].mean(), regime, color=mpl.colormaps.get_cmap('turbo')(cidx), ha='center', va='top')
    handles, _ = axes[0].get_legend_handles_labels()
    axes[0].legend([handles[0], (handles[1],handles[2])],
                ['Leader','Follower'],
                frameon=False, loc='upper left')

    return fig, axes



def regime_proportion(regime_list_HA, regime_list_HH):
    regime_list_HA = regime_list_HA[['Fa','Fd','C','A','D','F','S']].sum()
    regime_list_HA = regime_list_HA/regime_list_HA.sum()
    regime_list_HH = regime_list_HH[['Fa','Fd','C','A','D','F','S']].sum()
    regime_list_HH = regime_list_HH/regime_list_HH.sum()

    fig, ax = plt.subplots(1,1,figsize=(7.5,1.5))

    ax.bar(np.arange(0,21,3), regime_list_HH, fc=rgb_color('tab:blue',0.4), ec=rgb_color('tab:blue',1), label='HH')
    ax.bar(np.arange(0,21,3)+1, regime_list_HA, fc=rgb_color('tab:red',0.4), ec=rgb_color('tab:red',1), label='HA', ls='--')

    for x, regime in zip(np.arange(0,21,3), regime_list_HA.index):
        ax.text(x+1.05, regime_list_HA[regime]+0.005, str(round(regime_list_HA[regime]*100,1))+'%', ha='center', va='bottom', fontsize=8)
        ax.text(x+0.05, regime_list_HH[regime]+0.005, str(round(regime_list_HH[regime]*100,1))+'%', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(np.arange(0,21,3)+0.5)
    ax.set_xticklabels(['Fa','Fd','C','A','D','F','S'])

    ax.set_ylim(0,0.5)
    ax.set_yticks([])
    ax.set_xlabel('Car-following regimes')

    ax.legend(frameon=False, loc='upper left')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return fig, ax



def headway_dist_regime(cfdata_HH, cfdata_HA):
    fig, axes = plt.subplots(1,3,figsize=(5.5,1.3),gridspec_kw={'width_ratios':[1,0.05,1]})
    axes[1].axis('off')
    axes[0].set_xlabel('Min. THW in steady state (s)', labelpad=0.5)
    axes[0].set_ylabel('Probability density')
    axes[0].set_ylim([0.,1.25])
    axes[0].set_xticks([1,3,5])
    axes[0].set_yticks([0,0.5,1.])
    axes[2].set_xlabel('Follower speed (m/s)', labelpad=0.5)
    axes[2].set_ylabel('Min. THW \n in steady state (s)')
    axes[2].set_ylim([0.,6.5])
    axes[2].set_yticks([1,3,5])
    axes[2].set_xlim([1,24])
    axes[2].set_xticks([5,10,15,20])

    thw_bins = np.linspace(0, 6, 45)
    thw_range = np.linspace(0, 6, 100)

    thw_HH = cfdata_HH.loc[cfdata_HH.groupby('case_id')['thw'].idxmin()]
    thw_HH = thw_HH[thw_HH['regime'].isin(['F'])]
    thw_HA = cfdata_HA.loc[cfdata_HA.groupby('case_id')['thw'].idxmin()]
    thw_HA = thw_HA[thw_HA['regime'].isin(['F'])]

    plot_headway_subplot(axes[0], thw_HH['thw'], thw_bins, thw_range, 'tab:blue', 'HH', 0.95, '-', method=0.2)
    plot_headway_subplot(axes[0], thw_HA['thw'], thw_bins, thw_range, 'tab:red', 'HA', -0.9, '--', method=0.2)
    # pvalue, result = ks_test(thw_HH['thw'], thw_HA['thw'])
    # axes[0].text(0.99, 0.53, f'K-S test p-value: {pvalue:.3f}', transform=axes[0].transAxes, va='center', ha='right')
    # axes[0].text(0.99, 0.4, result, transform=axes[0].transAxes, va='center', ha='right')
    axes[2].plot(thw_HH['v_follower'], thw_HH['thw'], 's', ms=1.5, fillstyle='none', mew=0.2, c='tab:blue', label='HH', rasterized=True)
    axes[2].plot(thw_HA['v_follower'], thw_HA['thw'], 'o', ms=1.5, fillstyle='none', mew=0.2, c='tab:red', label='HA', rasterized=True)
    
    handles, _ = axes[0].get_legend_handles_labels()
    axes[0].legend([(handles[0],handles[1]),(handles[2],handles[3])],
                   ['HH','HA'],
                   frameon=False, loc='upper right')
    
    axes[2].legend([Line2D([], [], color='none', marker='s', markerfacecolor='none', markeredgecolor=rgb_color('tab:blue'), markersize=3),
                    Line2D([], [], color='none', marker='o', markerfacecolor='none', markeredgecolor=rgb_color('tab:red'), markersize=3)],
                   ['HH','HA'],
                   frameon=False, loc='upper right', handletextpad=0.05, labelspacing=0.1)
    
    return fig, axes



def idm_parameters(idm_HH, idm_HA):
    vars = ['v_0','s_0','T','alpha','beta']
    titles = [r'$v_0$ $(m/s)$',
              r'$s_0$ $(m)$',
              r'$T$ $(s)$',
              r'$\alpha$ $(m/s^2)$',
              r'$\beta$ $(m/s^2)$']

    n_bins = 20

    bins = [np.linspace(10-(29-10)/(n_bins-1)/2,29+(29-10)/(n_bins-1)/2,n_bins), 
            np.linspace(3.5-(16-3.5)/(n_bins-1)/2,16+(16-3.5)/(n_bins-1)/2,n_bins),  
            np.linspace(0.-(5-0.)/(n_bins-1)/2,5+(5-0.)/(n_bins-1)/2,n_bins), 
            np.linspace(0.3-(6.-0.3)/(n_bins-1)/2,6.+(6.-0.3)/(n_bins-1)/2,n_bins), 
            np.linspace(0.3-(6.-0.3)/(n_bins-1)/2,6.+(6.-0.3)/(n_bins-1)/2,n_bins)]
    
    xticks = [[12,18,24],
              [5,9,13],
              [1,2,3,4],
              [1,3,5],
              [1,3,5]]

    fig, axes = plt.subplots(1,5,figsize=(7.5,0.9),sharey=True)
    for var, title, bin, xtick, ax in zip(vars, titles, bins, xticks, axes):
        for results,label,color,ls in zip([idm_HH, idm_HA], ['HH','HA'],['tab:blue','tab:red'],['-','--']):
            ax.hist(results[var], bins=bin,
                    weights=np.ones(len(results[var]))/len(results[var]),
                    fc=rgb_color(color,0.3), ec=rgb_color(color,1),histtype='stepfilled', ls=ls,
                    linewidth=0.5, label=label)
        ax.set_xlabel(title, labelpad=1.5, fontsize=8)
        ax.set_xticks(xtick)
        table = ax.table(cellText=np.round([[idm_HH[var].mean(), idm_HH[var].std()],
                                            [idm_HA[var].mean(), idm_HA[var].std()]],2),
                colLabels=['Mean','Std.'],
                loc='lower center',
                bbox=[0.,-1.3,1.,0.65],
                cellLoc='center',
                rowLoc='center',
                edges='horizontal')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for child in table.get_children():
            child.set(linewidth=0.5)
        for cell in table._cells:
            text = table._cells[cell].get_text()
            text.set_fontfamily('Times New Roman')

        # kstest = ks_2samp(idm_HA[var], idm_HH[var])
        # pvalue = "{:.3f}".format(kstest.pvalue)
        # if kstest.pvalue<0.01:
        #     same = 'False'
        # else:
        #     same = 'True'
        # table = ax.table(cellText=[[same,pvalue]],
        #                  colLabels=['Same','$p$'],
        #                  loc='lower center',
        #                  bbox=[0.,-1.85,1.,0.42],
        #                  cellLoc='center',
        #                  rowLoc='center',
        #                  edges='horizontal')
        # table.auto_set_font_size(False)
        # table.set_fontsize(9)
        # for child in table.get_children():
        #     child.set(linewidth=0.5)
        # for cell in table._cells:
        #     text = table._cells[cell].get_text()
        #     text.set_fontfamily('Times New Roman')

    axes[0].set_ylabel('Relative frequency', y=0.4, fontsize=8)
    handle, label = axes[0].get_legend_handles_labels()
    axes[0].legend([handle[0], handle[1]], 
                   ['HH','HA'], 
                   loc='lower center',
                   handletextpad=-1.7,
                   bbox_to_anchor=(-0.2, -1.425),
                   handlelength=2,
                   handleheight=1.1,
                   frameon=False, fontsize=8)
    
    # axes[0].text(5, -0.86, 'K-S test', ha='center', va='center', fontsize=8)
    
    return fig, axes



def gipps_parameters(gipps_HH, gipps_HA):
    vars = ['v_0', 's_0', 'tau', 'alpha', 'b', 'b_leader']
    titles = [r'$v_0$ $(m/s)$',
              r'$s_0$ $(m)$',
              r'$tau$ $(s)$',
              r'$\alpha$ $(m/s^2)$',
              r'$b$ $(m/s^2)$',
              r'$b_{\mathrm{leader}}$ $(m/s^2)$']

    n_bins = 20

    bins = [np.linspace(10-(29-10)/(n_bins-1)/2,29+(29-10)/(n_bins-1)/2,n_bins), 
            np.linspace(3.5-(16-3.5)/(n_bins-1)/2,16+(16-3.5)/(n_bins-1)/2,n_bins),  
            np.linspace(0.-(5-0.)/(n_bins-1)/2,5+(5-0.)/(n_bins-1)/2,n_bins), 
            np.linspace(0.3-(6.-0.3)/(n_bins-1)/2,6.+(6.-0.3)/(n_bins-1)/2,n_bins), 
            np.linspace(0.3-(6.-0.3)/(n_bins-1)/2,6.+(6.-0.3)/(n_bins-1)/2,n_bins),
            np.linspace(0.3-(6.-0.3)/(n_bins-1)/2,6.+(6.-0.3)/(n_bins-1)/2,n_bins)]
    
    xticks = [[12,18,24],
              [5,9,13],
              [1,2,3,4],
              [1,3,5],
              [1,3,5],
              [1,3,5]]

    fig, axes = plt.subplots(1,6,figsize=(7.5,0.9),sharey=True)
    for var, title, bin, xtick, ax in zip(vars, titles, bins, xticks, axes):
        for results,label,color,ls in zip([gipps_HH, gipps_HA], ['HH','HA'],['tab:blue','tab:red'],['-','--']):
            ax.hist(results[var], bins=bin,
                    weights=np.ones(len(results[var]))/len(results[var]),
                    fc=rgb_color(color,0.3), ec=rgb_color(color,1),histtype='stepfilled', ls=ls,
                    linewidth=0.5, label=label)
        ax.set_xlabel(title, labelpad=1.5, fontsize=8)
        ax.set_xticks(xtick)
        table = ax.table(cellText=np.round([[gipps_HH[var].mean(), gipps_HH[var].std()],
                                            [gipps_HA[var].mean(), gipps_HA[var].std()]],2),
                colLabels=['Mean','Std.'],
                loc='lower center',
                bbox=[0.,-1.3,1.,0.65],
                cellLoc='center',
                rowLoc='center',
                edges='horizontal')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for child in table.get_children():
            child.set(linewidth=0.5)
        for cell in table._cells:
            text = table._cells[cell].get_text()
            text.set_fontfamily('Times New Roman')

    axes[0].set_ylabel('Relative frequency', y=0.4, fontsize=8)
    handle, label = axes[0].get_legend_handles_labels()
    axes[0].legend([handle[0], handle[1]], 
                   ['HH','HA'], 
                   loc='lower center',
                   handletextpad=-1.7,
                   bbox_to_anchor=(-0.2, -1.425),
                   handlelength=2,
                   handleheight=1.1,
                   frameon=False, fontsize=8)
    
    return fig, axes



def idm_sampling(idm_HA, idm_HH):
    vars = ['v_0','s_0','T','alpha','beta']
    titles = [r'$v_0$ $(m/s)$',
              r'$s_0$ $(m)$',
              r'$T$ $(s)$',
              r'$\alpha$ $(m/s^2)$',
              r'$\beta$ $(m/s^2)$']

    n_bins = 20

    bins = [np.linspace(10-(29-10)/(n_bins-1)/2,29+(29-10)/(n_bins-1)/2,n_bins), 
            np.linspace(3.5-(16-3.5)/(n_bins-1)/2,16+(16-3.5)/(n_bins-1)/2,n_bins),  
            np.linspace(0.-(5-0.)/(n_bins-1)/2,5+(5-0.)/(n_bins-1)/2,n_bins), 
            np.linspace(0.3-(6.-0.3)/(n_bins-1)/2,6.+(6.-0.3)/(n_bins-1)/2,n_bins), 
            np.linspace(0.3-(6.-0.3)/(n_bins-1)/2,6.+(6.-0.3)/(n_bins-1)/2,n_bins)]
    
    xticks = [[12,18,24],
              [5,9,13],
              [1,2,3,4],
              [1,3,5],
              [1,3,5]]

    fig, axes = plt.subplots(1,5,figsize=(7.5,0.9),sharey=True)
    for var, title, bin, xtick, ax in zip(vars, titles, bins, xticks, axes):
        for results,label,color,ls in zip([idm_HA, idm_HH], ['HA','HH'],['blue','tab:blue'], ['--','-']):
            ax.hist(results[var], bins=bin,
                    weights=np.ones(len(results[var]))/len(results[var]),
                    fc=rgb_color(color,0.3), ec=rgb_color(color,1),histtype='stepfilled', ls=ls,
                    linewidth=0.5, label=label)
        ax.set_xlabel(title, labelpad=1.5, fontsize=8)
        ax.set_xticks(xtick)
        table = ax.table(cellText=np.round([[idm_HA[var].mean(), idm_HA[var].std()],
                                            [idm_HH[var].mean(), idm_HH[var].std()]],2),
                colLabels=['Mean','Std.'],
                loc='lower center',
                bbox=[0.,-1.3,1.,0.65],
                cellLoc='center',
                rowLoc='center',
                edges='horizontal')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for child in table.get_children():
            child.set(linewidth=0.5)
        for cell in table._cells:
            text = table._cells[cell].get_text()
            text.set_fontfamily('Times New Roman')

        # kstest = ks_2samp(idm_HA[var], idm_HH[var])
        # pvalue = "{:.3f}".format(kstest.pvalue)
        # if kstest.pvalue<0.01:
        #     same = 'False'
        # else:
        #     same = 'True'
        # table = ax.table(cellText=[[same,pvalue]],
        #                  colLabels=['Same','$p$'],
        #                  loc='lower center',
        #                  bbox=[0.,-1.85,1.,0.42],
        #                  cellLoc='center',
        #                  rowLoc='center',
        #                  edges='horizontal')
        # table.auto_set_font_size(False)
        # table.set_fontsize(9)
        # for child in table.get_children():
        #     child.set(linewidth=0.5)
        # for cell in table._cells:
        #     text = table._cells[cell].get_text()
        #     text.set_fontfamily('Times New Roman')

    axes[0].set_ylabel('Relative frequency', y=0.4, fontsize=8)
    handle, label = axes[0].get_legend_handles_labels()
    fig.legend([handle[0], handle[1]], 
                ['291 sampled HH followers','1,207 other HH followers'], 
                loc='upper center', ncol=2,
                # handletextpad=-1.7,
                bbox_to_anchor=(0.52, 1.2), 
                frameon=False)
    
    # axes[0].text(5, -0.86, 'K-S test', ha='center', va='center', fontsize=8)
    
    return fig, axes



def headway_leader_variability(cfdata_HH, cfdata_HA, count):
    fig, axes = plt.subplots(1,3,figsize=(5.5,1.3),gridspec_kw={'width_ratios':[1,0.05,1]})
    axes[1].axis('off')
    axes[0].set_xlabel('Min. THW in steady state (s)', labelpad=0.5)
    axes[0].set_ylabel('Probability density')
    axes[0].set_ylim([0.,1.25])
    axes[0].set_xticks([1,3,5])
    axes[0].set_yticks([0,0.5,1.])
    axes[2].set_xlabel('Follower speed (m/s)', labelpad=0.5)
    axes[2].set_ylabel('Min. THW \n in steady state (s)')
    axes[2].set_ylim([0.,6.5])
    axes[2].set_yticks([1,3,5])
    axes[2].set_xlim([1,24])
    axes[2].set_xticks([5,10,15,20])

    thw_bins = np.linspace(0, 6, 45)
    thw_range = np.linspace(0, 6, 100)

    thw_HH = cfdata_HH.loc[cfdata_HH.groupby('case_id')['thw'].idxmin()]
    thw_HH = thw_HH[thw_HH['regime'].isin(['F'])]
    thw_HA = cfdata_HA.loc[cfdata_HA.groupby('case_id')['thw'].idxmin()]
    thw_HA = thw_HA[thw_HA['regime'].isin(['F'])]

    if count in [0,2]:
        thwHH = 1.0
        thwHA = -0.9
    if count==3:
        thwHH = -0.95
        thwHA = 1.0
    if count==4:
        thwHH = 1.0
        thwHA = -1.0
    plot_headway_subplot(axes[0], thw_HH['thw'], thw_bins, thw_range, 'tab:blue', 'HH', thwHH, '-', method=0.15)
    plot_headway_subplot(axes[0], thw_HA['thw'], thw_bins, thw_range, 'tab:red', 'HA', thwHA, '--', method=0.2)
    pvalue, result = ks_test(thw_HH['thw'], thw_HA['thw'], text1='Heterogeneous\n', text2='Uniform')
    axes[0].text(0.99, 0.88, f'K-S p-value: {pvalue:.3f}', transform=axes[0].transAxes, va='center', ha='right')
    axes[0].text(0.99, 0.65, result, transform=axes[0].transAxes, va='center', ha='right')
    axes[2].plot(thw_HH['v_follower'], thw_HH['thw'], 's', ms=1.5, fillstyle='none', mew=0.2, c='tab:blue', label='HH', rasterized=True)
    axes[2].plot(thw_HA['v_follower'], thw_HA['thw'], 'o', ms=1.5, fillstyle='none', mew=0.2, c='tab:red', label='HA', rasterized=True)

    axes[1].set_title('The same HV followers in HH following', fontsize=8, pad=10, x=0.45)
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend([(handles[0],handles[1]),(handles[2],handles[3]),
                Line2D([], [], color='none', marker='s', markerfacecolor='none', markeredgecolor=rgb_color('tab:blue'), markersize=3),
                Line2D([], [], color='none', marker='o', markerfacecolor='none', markeredgecolor=rgb_color('tab:red'), markersize=3)],
               ['1,497 leaders in HH','1 leader in HH','1,497 leaders in HH','1 leader in HH'],
               frameon=False, bbox_to_anchor=(0.5, -0.3), ncol=4, loc='lower center',
               columnspacing=1, handletextpad=0.2, borderpad=0.2, handlelength=1.6)

    return fig, axes



def headway_follower_variability(cfdata_HH, cfdata_HA):
    fig, axes = plt.subplots(1,3,figsize=(5.5,1.3),gridspec_kw={'width_ratios':[1,0.05,1]})
    axes[1].axis('off')
    axes[0].set_xlabel('Min. THW in steady state (s)', labelpad=0.5)
    axes[0].set_ylabel('Probability density')
    axes[0].set_ylim([0.,0.95])
    axes[0].set_xticks([1,3,5])
    axes[0].set_yticks([0,0.4,0.8])
    axes[2].set_xlabel('Follower speed (m/s)', labelpad=0.5)
    axes[2].set_ylabel('Min. THW \n in steady state (s)')
    axes[2].set_ylim([0.,6.5])
    axes[2].set_yticks([1,3,5])
    axes[2].set_xlim([1,24])
    axes[2].set_xticks([5,10,15,20])

    thw_bins = np.linspace(0, 6, 45)
    thw_range = np.linspace(0, 6, 100)

    thw_HH = cfdata_HH.loc[cfdata_HH.groupby('case_id')['thw'].idxmin()]
    thw_HH = thw_HH[thw_HH['regime'].isin(['F'])]
    thw_HA = cfdata_HA.loc[cfdata_HA.groupby('case_id')['thw'].idxmin()]
    thw_HA = thw_HA[thw_HA['regime'].isin(['F'])]

    plot_headway_subplot(axes[0], thw_HH['thw'], thw_bins, thw_range, 'tab:blue', 'HH', 1.05, '-', method=0.1)
    plot_headway_subplot(axes[0], thw_HA['thw'], thw_bins, thw_range, 'tab:red', 'HA', -0.95, '--', method=0.1)
    pvalue, result = ks_test(thw_HH['thw'], thw_HA['thw'])
    axes[0].text(0.99, 0.88, f'K-S test p-value: {pvalue:.3f}', transform=axes[0].transAxes, va='center', ha='right')
    axes[0].text(0.99, 0.75, result, transform=axes[0].transAxes, va='center', ha='right')
    axes[2].plot(thw_HH['v_follower'], thw_HH['thw'], 's', ms=1.5, fillstyle='none', mew=0.2, c='tab:blue', label='HH', rasterized=True)
    axes[2].plot(thw_HA['v_follower'], thw_HA['thw'], 'o', ms=1.5, fillstyle='none', mew=0.2, c='tab:red', label='HA', rasterized=True)

    axes[1].set_title('The same HV leaders in HH followed by', fontsize=8, pad=10, x=0.45)
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend([(handles[0],handles[1]),(handles[2],handles[3]),
                Line2D([], [], color='none', marker='s', markerfacecolor='none', markeredgecolor=rgb_color('tab:blue'), markersize=3),
                Line2D([], [], color='none', marker='o', markerfacecolor='none', markeredgecolor=rgb_color('tab:red'), markersize=3)],
               ['1,207 followers in HH','291 followers in HH','1,207 followers in HH','291 followers in HH'],
               frameon=False, bbox_to_anchor=(0.5, -0.3), ncol=4, loc='lower center',
               columnspacing=1, handletextpad=0.2, borderpad=0.2, handlelength=1.6)

    return fig, axes



def evaluate_classifier(confusion_matrices, loss_records):
    confusion_matrices['accuracy'] = (confusion_matrices['TP']+confusion_matrices['TN'])/confusion_matrices['size']
    confusion_matrices['precision'] = confusion_matrices['TP']/(confusion_matrices['TP']+confusion_matrices['FP'])
    confusion_matrices['recall'] = confusion_matrices['TP']/(confusion_matrices['TP']+confusion_matrices['FN'])
    confusion_matrices['F1'] = 2*confusion_matrices['precision']*confusion_matrices['recall']/(confusion_matrices['precision']+confusion_matrices['recall'])
    
    fig, axes = plt.subplots(1,3,figsize=(5.5,1.2),sharex=True,constrained_layout=True)
    loss_records = loss_records.loc[np.arange(5,205,5)-1]
    axes[0].plot(loss_records.index+1,loss_records.mean(axis=1),lw=0.5,c='tab:blue',label='Train')
    axes[0].scatter(loss_records.index+1,loss_records.mean(axis=1),s=10,ec='tab:blue',fc='none',lw=0.5,marker='o',label='Train')
    colors = ['tab:blue','tab:orange','tab:green']
    for index,col,marker in zip(['train','val','test'],colors,['o','d','s']):
        confusion_matrix = confusion_matrices[confusion_matrices['index']==index]
        axes[1].plot(confusion_matrix['num_epoches'], confusion_matrix['accuracy'], c=col, lw=0.5, label='accuracy')
        axes[1].scatter(confusion_matrix['num_epoches'], confusion_matrix['accuracy'], ec=col, fc='none', lw=0.5, marker=marker, s=10, label='accuracy')
        axes[2].plot(confusion_matrix['num_epoches'], confusion_matrix['F1'], c=col, lw=0.5, label='F1')
        axes[2].scatter(confusion_matrix['num_epoches'], confusion_matrix['F1'], ec=col, fc='none', lw=0.5, marker=marker, s=10, label='F1')
    for ax in [axes[1],axes[2]]:
        ax.fill_betweenx([0.4,1],142,148,fc=rgb_color('tab:red',0.2),ec=None)
    axes[0].set_title('Average loss', fontsize=8, pad=2)
    axes[1].set_title('Accuracy', fontsize=8, pad=2)
    axes[2].set_title('F1 score', fontsize=8, pad=2)
    axes[0].set_xlabel('Epoch', labelpad=0.5)
    axes[1].set_xlabel('Epoch', labelpad=0.5)
    axes[2].set_xlabel('Epoch', labelpad=0.5)
    axes[0].set_xticks(np.arange(0,250,50))
    axes[0].set_ylim([0.05,0.6])
    axes[1].set_ylim([0.4,1])
    axes[1].set_yticks([0.6,0.7,0.8,0.9])
    axes[2].set_ylim([0.4,1])
    axes[2].set_yticks([0.6,0.7,0.8,0.9])
    handle1, label1 = axes[0].get_legend_handles_labels()
    handle2, label2 = axes[1].get_legend_handles_labels()
    handle3, label3 = axes[2].get_legend_handles_labels()
    axes[0].legend([(handle1[0], handle1[1])],['Train'],frameon=False,ncol=1,loc='upper right')
    axes[2].legend([(handle2[2], handle2[3]),(handle2[4], handle2[5])], 
                   ['Val.','Test'], 
                   frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.05))

    return fig, axes



def scatter_zero_test(zero_tests,epoches):
    fig, axes = plt.subplots(1,4,figsize=(7.,1.2),sharex=True,sharey=True,constrained_layout=True)
    for axid,zero_test,epoch in zip([0,1,2,3],zero_tests,epoches):
        axes[axid].plot([0.015,0.015],[0,1],c='tab:gray',ls='--',lw=0.5, alpha=0.5)
        axes[axid].plot([-0.005,0.105],[0.5,0.5],c='tab:gray',ls='--',lw=0.5, alpha=0.5)
        axes[axid].scatter(zero_test['std'], zero_test['pred'], s=12, ec='tab:red', alpha=0.7, fc='none', lw=0.7, marker='+')
        axes[axid].set_title('Epoch '+str(epoch), fontsize=8, pad=2)
    fig.text(0.53, -0.07, 'Added noise', ha='center', fontsize=8)
    axes[0].set_ylabel('Classification')
    axes[0].set_ylim([-0.01,1.02])
    axes[0].set_yticks([0,0.5,1])
    axes[0].set_yticklabels(['HV','0.5','AV'])
    axes[0].set_xlim([-0.005,0.105])
    return fig, axes



def compare_leader(cfdata_likeHV, cfdata_likeAV, cfdata_HH):
    fig, axes = plt.subplots(1,5,figsize=(7.5,1.1),constrained_layout=True,gridspec_kw={'width_ratios': [1,1,0.02,1,1]})
    axes[2].axis('off')
    
    sample_list = [cfdata_likeHV[cfdata_likeHV['regime'].isin(['C','F'])].groupby(['case_id','regime'])['v_leader'].max(),
                   cfdata_likeAV[cfdata_likeAV['regime'].isin(['C','F'])].groupby(['case_id','regime'])['v_leader'].max(),
                   cfdata_likeHV[cfdata_likeHV['regime'].isin(['C','F'])].groupby(['case_id','regime'])['v_leader'].max(),
                   cfdata_HH[cfdata_HH['regime'].isin(['C','F'])].groupby(['case_id','regime'])['v_leader'].max(),
                   cfdata_likeHV[cfdata_likeHV['regime'].isin(['A','D','Fa','Fd'])].groupby(['case_id','regime'])['a_leader'].agg(['max','min']).values.flatten(),
                   cfdata_likeAV[cfdata_likeAV['regime'].isin(['A','D','Fa','Fd'])].groupby(['case_id','regime'])['a_leader'].agg(['max','min']).values.flatten(),
                   cfdata_likeHV[cfdata_likeHV['regime'].isin(['A','D','Fa','Fd'])].groupby(['case_id','regime'])['a_leader'].agg(['max','min']).values.flatten(),
                   cfdata_HH[cfdata_HH['regime'].isin(['A','D','Fa','Fd'])].groupby(['case_id','regime'])['a_leader'].agg(['max','min']).values.flatten()]
    # sample_list = [cfdata_likeHV.groupby(['case_id'])['v_leader'].max(),
    #                cfdata_likeAV.groupby(['case_id'])['v_leader'].max(),
    #                cfdata_likeHV.groupby(['case_id'])['v_leader'].max(),
    #                cfdata_HH.groupby(['case_id'])['v_leader'].max(),
    #                cfdata_likeHV.groupby(['case_id'])['a_leader'].agg(['max','min']).values.flatten(),
    #                cfdata_likeAV.groupby(['case_id'])['a_leader'].agg(['max','min']).values.flatten(),
    #                cfdata_likeHV.groupby(['case_id'])['a_leader'].agg(['max','min']).values.flatten(),
    #                cfdata_HH.groupby(['case_id'])['a_leader'].agg(['max','min']).values.flatten()]
    ax_list = [axes[0], axes[0],
               axes[1], axes[1],
               axes[3], axes[3],
               axes[4], axes[4]]
    bin_list = [np.linspace(0,24,24), np.linspace(0,24,24),
                np.linspace(0,24,24), np.linspace(0,24,24),
                np.linspace(-5,5,24), np.linspace(-5,5,24),
                np.linspace(-5,5,24), np.linspace(-5,5,24)]
    label_list = ['AV driving like HV', 'AV driving like AV',
                  'AV driving like HV', 'Real HV',
                  'AV driving like HV', 'AV driving like AV',
                  'AV driving like HV', 'Real HV']
    fc_list = [rgb_color('tab:blue',0.2), rgb_color('tab:red',0.2),
               rgb_color('tab:blue',0.2), rgb_color('tab:orange',0.2),
               rgb_color('tab:blue',0.2), rgb_color('tab:red',0.2),
               rgb_color('tab:blue',0.2), rgb_color('tab:orange',0.2)]
    ec_list = [rgb_color('tab:blue',1), rgb_color('tab:red',1),
               rgb_color('tab:blue',1), rgb_color('tab:orange',1),
               rgb_color('tab:blue',1), rgb_color('tab:red',1),
               rgb_color('tab:blue',1), rgb_color('tab:orange',1)]
    ls_list = ['-','--',
               '-','-.',
               '-','--',
               '-','-.']

    for sample, ax, bins, label, fc, ec, ls in zip(sample_list, ax_list, bin_list, label_list, fc_list, ec_list, ls_list):
        ax.hist(sample, bins=bins, density=True,
                histtype='stepfilled', fc=fc, ec=ec, lw=0.8, ls=ls, label=label)
        # ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(-0.02,1.04), handlelength=1.2, handletextpad=0.2)

    axes[0].set_ylabel('Probability density')
    # for idx,title in zip(range(2),['Classified HV vs. classified AV','Classified HV vs. real HV']):
    #     axes[idx].set_title(title, fontsize=8, pad=0)
    #     axes[idx+3].set_title(title, fontsize=8, pad=0)

    ax = fig.add_axes([0.06, 0.0, 0.44, 0.96], frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Maximum speed (m/s)', labelpad=1.5)
    
    ax = fig.add_axes([0.56, 0.0, 0.44, 0.96], frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Maximum acceleration and deceleration (m/s$^2$)', labelpad=-0.5)

    handle1, label1 = axes[0].get_legend_handles_labels()
    handle2, label2 = axes[1].get_legend_handles_labels()
    fig.legend([handle1[0],handle1[1],handle2[1]],
               [label1[0],label1[1],label2[1]],
               frameon=False, loc='lower center', ncol=3, bbox_to_anchor=(0.53,0.92))  

    for ax in [axes[0],axes[1]]:
        ax.set_ylim([0,0.28])
        ax.set_yticks([])
    axes[0].set_yticks([0,0.1,0.2])
    for ax in [axes[4],axes[3]]:
        ax.set_ylim([0,1.04])
        ax.set_yticks([])
    axes[3].set_yticks([0,0.4,0.8])  

    return fig, axes



def headway_dist_dynamics(cfdata, caseORtime='pred_case'):
    fig, axes = plt.subplots(1,3,figsize=(5.5,1.3),gridspec_kw={'width_ratios':[1,0.05,1]})
    axes[1].axis('off')
    axes[0].set_xlabel('Min. THW in steady state (s)', labelpad=0.5)
    axes[0].set_ylabel('Probability density')
    axes[0].set_ylim([0.,1.25])
    axes[0].set_xticks([1,3,5])
    axes[0].set_yticks([0,0.5,1])
    axes[2].set_xlabel('Follower speed (m/s)', labelpad=0.5)
    axes[2].set_ylabel('Min. THW \n in steady state (s)')
    axes[2].set_ylim([0.,6.5])
    axes[2].set_yticks([1,3,5])
    axes[2].set_xlim([1,19])
    axes[2].set_xticks([5,10,15])

    thw_bins = np.linspace(0, 6, 45)
    thw_range = np.linspace(0, 6, 100)

    thw = cfdata.loc[cfdata.groupby('case_id')['thw'].idxmin()]
    thw_HH = thw[(thw[caseORtime]<0.5)&(thw['regime'].isin(['F']))]
    thw_HA = thw[(thw[caseORtime]>0.5)&(thw['regime'].isin(['F']))]

    if caseORtime=='pred_case':
        thwHA = -0.9
        thwHH = 0.95
        title = 'Case-based classification according to $p_c$'
        labels = ['AV leader case classified as HV','AV leader case classified as AV']
    else:
        thwHA = -0.9
        thwHH = 0.92
        title = 'Operation-based classification according to $p_c^{(t)}$'
        labels = ['AV leader operation classified as HV','AV leader operation classified as AV']
    plot_headway_subplot(axes[0], thw_HH['thw'], thw_bins, thw_range, 'tab:blue', 'HH', thwHH, '-', method=0.3)
    plot_headway_subplot(axes[0], thw_HA['thw'], thw_bins, thw_range, 'tab:red', 'HA', thwHA, '--', method=0.3)
    pvalue, result = ks_test(thw_HH['thw'], thw_HA['thw'], text1='HV-like', text2='AV-like')
    axes[0].text(0.99, 0.88, f'K-S test p-value: {pvalue:.3f}', transform=axes[0].transAxes, va='center', ha='right')
    axes[0].text(0.99, 0.75, result, transform=axes[0].transAxes, va='center', ha='right')
    axes[2].plot(thw_HA['v_follower'], thw_HA['thw'], 'o', ms=1.5, fillstyle='none', mew=0.2, c=rgb_color('tab:red',0.6), label='HA', rasterized=True)
    axes[2].plot(thw_HH['v_follower'], thw_HH['thw'], 's', ms=1.5, fillstyle='none', mew=0.3, c='tab:blue', label='HH', rasterized=True)
        
    axes[1].set_title(title, fontsize=8, pad=10, x=0.45)
    handles, _ = axes[0].get_legend_handles_labels()
    axes[0].legend([(handles[0],handles[1]),(handles[2],handles[3])], labels,
                   frameon=False, bbox_to_anchor=(0.5, -0.75), ncol=1, loc='lower center',
                   columnspacing=1, handletextpad=0.2, borderpad=0.2, handlelength=1.6)
    
    axes[2].legend([Line2D([], [], color='none', marker='s', markerfacecolor='none', markeredgecolor=rgb_color('tab:blue'), markersize=3),
                    Line2D([], [], color='none', marker='o', markerfacecolor='none', markeredgecolor=rgb_color('tab:red'), markersize=3)],
                   labels,
                   frameon=False, bbox_to_anchor=(0.5, -0.75), ncol=1, loc='lower center',
                   columnspacing=1, handletextpad=0.2, borderpad=0.2, handlelength=1.6)

    return fig, axes