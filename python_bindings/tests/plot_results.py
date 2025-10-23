#!/usr/bin/env python3

import matplotlib.pyplot as plt
import argparse
import torch
import statistics
import os
import scipy

plt.rcParams["font.family"] = "Verdana"
plt.rcParams["font.sans-serif"] = "Verdana"

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('text', usetex=True)

fmt = 'C1--.'

dist_names = {
    'lognormal': 'LogNormal',
    'normal': 'Normal',
    'exponential': 'Exponential',
    'truncnorm': 'TruncNorm',
    'weibull': 'Weibull',
}

###############################################################################
###############################################################################

def saveFig(filename):
        
    plt.savefig('{}.pdf'.format(filename), dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format='pdf',
                transparent=False, bbox_inches='tight', pad_inches=0.1,
                metadata=None)

def plot_line(objectives, rounds, label, fmt, ax=None, linewidth=2, markersize=3, markevery=10, capsize=5, yerr=None):

    # Plot the data
    if ax is None:
        h = None
        if yerr:
            plt.errorbar(rounds, objectives, yerr=yerr, fmt=fmt, label=label, linewidth=linewidth, markersize=markersize,
                         markevery=markevery, capsize=capsize)
        else:
            plt.semilogy(rounds, objectives, fmt, label=label, linewidth=linewidth, markersize=markersize, markevery=markevery)

    else:
        if yerr is not None:
            h = ax.errorbar(rounds, objectives, yerr=yerr, fmt=fmt, label=label, linewidth=linewidth, markersize=markersize,
                         markevery=markevery, capsize=capsize)
        else:
            h, = ax.semilogy(rounds, objectives, fmt, label=label, linewidth=linewidth, markersize=markersize,
                         markevery=markevery)

    return h, label


def simba_update_lists(results, nmse_list, nmse_std_list, time_list, time_std_list, alg, d, s, q, hp=None):
    
    if results['nmse'][alg][dist][d][s][q][hp]:
        time_list.append(statistics.mean(results['sqv_time[ms]'][alg][dist][d][s][q][hp]))
        min_time_list.append(min(results['sqv_time[ms]'][alg][dist][d][s][q][hp]))
    
        if len(results['sqv_time[ms]'][alg][dist][d][s][q][hp]) > 1:
            time_std_list.append(scipy.stats.median_abs_deviation(results['sqv_time[ms]'][alg][dist][d][s][q][hp]))
        
        else:
            time_std_list.append(0)

        nmse_list.append(statistics.mean(results['nmse'][alg][dist][d][s][q][hp]))
        if len(results['nmse'][alg][dist][d][s][q][hp]) > 1:
            nmse_std_list.append(statistics.stdev(results['nmse'][alg][dist][d][s][q][hp]))
        else:
            nmse_std_list.append(0)

        return True
    
    else:
        return False

###############################################################################
###############################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""
    Plot result graphs of "Better than Optimal: Improving Adaptive Stochastic Quantization Using Shared Randomness" paper.
                     
    Execute the following to generate nmse/time vs. d graph: 
    $ python3 plot_results.py --results_fn ./results/results_lognormal.pt --show
        
    see more options by running: $  python3 plot_results.py --help
    
    """,
    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--results_fn', default='./results/results_lognormal.pt', help='exact results file path')
    
    parser.add_argument('--show', default=False, action='store_true', help='show graphs')

    args = parser.parse_args()

    ###############################################################################
    ###############################################################################

    titleFont = 16
    axisLabelsFont = 14
    axisTicksFont = 14
    legendFont = 15
    linewidth = 1
    markersize = 6
    ncols = 3


    # load results
    path = './results'
    os.makedirs(path, exist_ok=True)
    
    results = torch.load(args.results_fn, map_location='cpu') # exact results

    meta_data = results
        
    alg = meta_data['alg']
    if alg not in ['simba_exact']:
        raise ValueError('Invalid alg: {}, only simba_exact is supported'.format(alg))
    dist = meta_data['dist']
    vec_d_list = meta_data['vec_d_list']
    s = meta_data['s']
    l = meta_data['l']
    m_quiver = meta_data['m_quiver']
    hp = meta_data['hp']

    ###################################################################################
    ###################################################################################
            
    handles = []
    labels = []

    # define new 2x1 figure without space for a legend
    fig, ax = plt.subplots(2, 1, figsize=(4, 4))
    plt.subplots_adjust(wspace=.267, hspace=.35)
        
    nmse_list = []
    time_list = []
    min_time_list = []
    nmse_std_list = []
    time_std_list = []
    d_list = []
    
    for d in vec_d_list:
        # print('collect simba_exact results for alg={}, d={}, s={}, q={}, m={}'.format(alg, d, s, l, hp))
        valid = simba_update_lists(results, nmse_list, nmse_std_list, time_list, time_std_list, alg, d, s, l, hp=hp)
                
        if valid:
            d_list.append(d)

    alg_name = 'Simba ($\ell={}$)'.format(l)
    
    if nmse_list:
        handle, label = plot_line(nmse_list, d_list, alg_name, fmt,
                        ax=ax[0], linewidth=linewidth, markersize=markersize, markevery=1, yerr=nmse_std_list)

    if time_list:
        handle, label = plot_line(time_list, d_list, alg_name, fmt,
                            ax=ax[1], linewidth=linewidth, markersize=markersize, markevery=1, yerr=time_std_list)

    if 'l' in locals() and label not in labels:
        handles.append(handle)
        labels.append(label)
            
    ax[0].set_ylabel('vNMSE', fontsize=axisLabelsFont)
    ax[1].set_ylabel('Time [ms]', fontsize=axisLabelsFont)
    ax[1].set_xlabel(r'Dimension ($d$)', fontsize=axisLabelsFont)
    ax[0].set_xscale("log", base=2)
    ax[1].set_xscale("log", base=2)
    ax[1].set_yscale("log")
    ax[0].tick_params(labelsize=axisTicksFont)
    ax[1].tick_params(labelsize=axisTicksFont)
    ax[0].grid(linestyle='dashed')
    ax[1].grid(linestyle='dashed')

    fig.suptitle(r'{}: {} distribution, $s={}$, $m_{}={}$'.format(alg_name, dist_names[dist], s, '{quiver}', m_quiver), fontsize=titleFont)

    fn = '{}/d_s{}_m_{}_l_{}_{}'.format(path, s, m_quiver, l, dist)
    saveFig(fn)

    if args.show:
        plt.show()
    else:
        plt.close()

