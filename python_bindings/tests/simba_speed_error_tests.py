#!/usr/bin/env python3

import torch
import time
import argparse
from scipy.stats import truncnorm
import os
from datetime import datetime
import itertools

# our python package
import simba_cpp

##############################################################################
##############################################################################

def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

##############################################################################
##############################################################################

def quantize(vec, sqv):
    if (len(sqv.size()) == 1):
        buckets = torch.bucketize(vec, sqv) # buckets is the first that is larger or equal than!
       
        up = torch.take(sqv, buckets)
        down = torch.take(sqv, torch.clip(buckets - 1, min=0))
        
        p =  (up - vec) / (up - down)
        r = torch.rand(p.numel(), device=device)
        
        return down  + (up - down) * (p < r)
    else:    
        thresholds = calcThresholds(sqv)
        
        buckets = torch.bucketize(vec, thresholds)
        
        q = len(sqv)
        H = buckets % q 
        
        S = torch.div(buckets, q, rounding_mode='trunc')
        d = vec.numel()
        SR = torch.randint(q, (d,), device=device) 
        
        larger_than_SR  = SR < H
        smaller_than_SR = SR > H
        equal_to_SR     = SR == H
        nr_equal_to_SR = torch.sum(equal_to_SR)
        
        det_sum = torch.zeros(nr_equal_to_SR, device=device).double() 
        Sequal_to_SR = S[equal_to_SR]
        Hequal_to_SR = H[equal_to_SR]
        for H_prime in range(q):
            larger_than_H_prime  = H_prime < Hequal_to_SR
            smaller_than_H_prime = H_prime > Hequal_to_SR
            det_sum[larger_than_H_prime]  += sqv[H_prime][Sequal_to_SR[larger_than_H_prime]+1]
            det_sum[smaller_than_H_prime] += sqv[H_prime][Sequal_to_SR[smaller_than_H_prime]] 
        
        x_p = q*vec[equal_to_SR] - det_sum
        res = torch.zeros(d, device=device, dtype=torch.double)
        res[larger_than_SR]  = sqv[SR[larger_than_SR],S[larger_than_SR]+1]
        res[smaller_than_SR] = sqv[SR[smaller_than_SR],S[smaller_than_SR]]
        
        #s = torch.cuda.Event(enable_timing=True)        
        #e = torch.cuda.Event(enable_timing=True)
        #s.record()
        sqv_equal_to_SR_S = sqv[H,S][equal_to_SR]
        S_plus_1 = torch.clamp(S+1, max=sqv.size()[1]-1) # TODO: check BUG fix
        sqv_equal_to_SR_S_plus_1 = sqv[H,S_plus_1][equal_to_SR]
        sqv_equal_to_SR_S_plus_1___minus___sqv_equal_to_SR_S = sqv_equal_to_SR_S_plus_1 - sqv_equal_to_SR_S
        p_up = (x_p - sqv_equal_to_SR_S) / sqv_equal_to_SR_S_plus_1___minus___sqv_equal_to_SR_S
        r = torch.rand(nr_equal_to_SR, device=device)
        res[equal_to_SR] = sqv_equal_to_SR_S  + sqv_equal_to_SR_S_plus_1___minus___sqv_equal_to_SR_S * (r < p_up)
        
        return res

##############################################################################
##############################################################################

def calc_vNMSE(vec, sqv, snorm = None):
    if (len(sqv.size()) == 1):
        buckets = torch.bucketize(vec, sqv) # buckets is the first that is larger or equal than!
        
        up = sqv[buckets]#torch.take(sqv, buckets)
        down = sqv[buckets-1]#torch.take(sqv, torch.clip(buckets - 1, min=0))
        
        if snorm is None:
            return torch.sum((up-vec)*(vec-down)) / torch.norm(vec, 2) ** 2 
        return torch.sum((up-vec)*(vec-down)) / snorm 
    return calc_SR_vNMSE(vec, sqv, snorm)

##############################################################################
##############################################################################

def calcThresholds(sqv): #equivalent to return torch.Tensor([sum([sqv[j%q][j//q] for j in range(i,i+q)])/q  for i in range(1,q*(s-1))]).to(device)
    q, s = sqv.size()    
    thresholds = torch.zeros(q*(s-1), device=device)
    T = sum([sqv[j][0] for j in range(1,q)]) + sqv[0][1]
    thresholds[0] = T
    idx = 1
    for H in range(1,q):    
        T += sqv[H][1] - sqv[H][0]
        thresholds[idx] = T
        idx += 1
    for X in range(1, s-1):
        for H in range(0, q):
            T += sqv[H][X + 1] - sqv[H][X]
            thresholds[idx] = T
            idx += 1
    return thresholds/q

##############################################################################
##############################################################################

def calc_SR_vNMSE(vec, sqv, snorm = None):
    thresholds = calcThresholds(sqv)
    q, s = sqv.size()
    d = vec.size()
    buckets = torch.bucketize(vec, thresholds)
    H = buckets % q
    S = torch.div(buckets, q, rounding_mode='trunc') #S = buckets // q
    det_sum = torch.zeros(d, device=device).double()
    det_err_sum = torch.zeros(d, device=device).double()
    for H_prime in range(q):
        larger_than_H_prime = H_prime < H
        smaller_than_H_prime = H_prime > H
        det_sum[larger_than_H_prime]  += sqv[H_prime][S[larger_than_H_prime]+1]
        det_sum[smaller_than_H_prime] += sqv[H_prime][S[smaller_than_H_prime]]
        
        det_err_sum[larger_than_H_prime]  += (vec[larger_than_H_prime] - sqv[H_prime][S[larger_than_H_prime]+1])**2
        det_err_sum[smaller_than_H_prime] += (vec[smaller_than_H_prime] - sqv[H_prime][S[smaller_than_H_prime]])**2
    x_p = q*vec - det_sum
    S_plus_1 = torch.clamp(S+1, max=sqv.size()[1]-1) # TODO: check BUG fix
    p_up = (x_p - sqv[H,S]) / (sqv[H,S_plus_1] - sqv[H,S])
    p_up = torch.nan_to_num(p_up, nan=0, posinf=0, neginf=0)
    sq_err = p_up * (sqv[H,S_plus_1] - vec)**2 + (1-p_up)*(sqv[H,S] - vec)**2
    MSE = (det_err_sum + sq_err) / q
    if snorm is None:
        vMSE = torch.sum(MSE) / torch.norm(vec)**2
    else:
        vMSE = torch.sum(MSE) / snorm
    return vMSE

##############################################################################
##############################################################################

algs = {}

algs["simba_exact"] = {}
algs["simba_exact"]['alg'] = simba_cpp.simba
algs["simba_exact"]['description'] = "call: simba_cpp.simba(svec, s, q, iters, bin_iters, bin_iters_increase_threshold, stopping_threshold, m, debug, quantype, log_cost_fn); " + \
                                                            "svec is the sorted vector, s is the number of quantization values, q is the nubmer of shared random values, " +\
                                                            "iters is a bound on the number of iterations (moving all quantization values once), "+\
                                                            "bin_iters is a bound on the number of binary search iterations (for placing a single quantization value once), "+\
                                                            "bin_iters_increase_threshold is how much must the error decrease in each iteration bin_iters is increased, "+\
                                                            "stopping_threshold is a threshold that if we do not improve by the search halts, "+\
                                                            "m is the number of bins for the histogram (m=-1 means no histogram and using the original vector)" +\
                                                            "debug enables debug prints" +\
                                                            "quantype is the type of discretization ('Histogram', 'Uniform', 'Quantiles') "

##############################################################################
##############################################################################
def eval(alg, vec, results, d, s, l=None, iters=None, bin_iters=None, bin_iters_increase_threshold=None, stopping_threshold=None, m_quiver=None, is_log_cost_fn=False, debug=False):
    if torch.cuda.is_available():
        start_gpu = torch.cuda.Event(enable_timing=True)
        end_gpu = torch.cuda.Event(enable_timing=True)
        start_gpu.record()

    ### sort vector and move to cpu
    svec, _ = torch.sort(vec)
    svec = svec.double().to('cpu')

    if torch.cuda.is_available():
        end_gpu.record()
        torch.cuda.synchronize()
        results['sort_time[ms]'][alg][args.dist][d][s][l][hp].append(start_gpu.elapsed_time(end_gpu))
    else:
        results['sort_time[ms]'][alg][args.dist][d][s][l][hp].append(None)
        
    try:
        
        # measure init time for simba_exact:
        if alg == 'simba_exact':        
            ### invoke and time
            start = time.time_ns()

            if is_log_cost_fn:
                seed_num = len(results['sqv_time[ms]'][alg][args.dist][d][s][l][hp])
                log_cost_fn = './results/' + args.dist + '_logs/log_cost_{}_d{}_s{}_q{}_m{}_seed{}.log'.format(args.dist, d, s, l, m_quiver, seed_num)
            else:
                log_cost_fn = ""
            
            sqv = algs[alg]['alg'](svec, s, l, iters, bin_iters, bin_iters_increase_threshold, stopping_threshold, m_quiver, False, log_cost_fn)
            
            end = time.time_ns()
        
            results['sqv_time[ms]'][alg][args.dist][d][s][l][hp].append((end - start)/1000000)

        else:
            raise RuntimeError('Receieved {} alg for evaluation. However, this script supports only \'simba_exact\' alg'.format(alg))
        
        if torch.cuda.is_available():
            start_gpu.record()

        dvec = quantize(vec, sqv.to(device))

        if torch.cuda.is_available():
            end_gpu.record()
            torch.cuda.synchronize()
            results['quantize_time[ms]'][alg][args.dist][d][s][l][hp].append(start_gpu.elapsed_time(end_gpu))
        else:
            results['quantize_time[ms]'][alg][args.dist][d][s][l][hp].append(None)

        nmse = calc_vNMSE(vec.to(device), sqv.to(device))
        results['nmse'][alg][args.dist][d][s][l][hp].append(nmse.item())
        
        if args.verbose:
            print('({}){}, {}, d={}, s={}, l={}, hp={}: nmse={:.5f}, time[ms]={:.2f}'.format(seed_idx+1, alg, args.dist, d, s, l, hp, nmse, (end - start)/1000000))

    except Exception as error:
        print('Error:', error)
        results['sqv_time[ms]'][alg][args.dist][d][s][l][hp].append(None)
        results['quantize_time[ms]'][alg][args.dist][d][s][l][hp].append(None)
        results['nmse'][alg][args.dist][d][s][l][hp].append(None)
        
        if args.verbose:
            print('{}, {}, d={}, s={}, l={}, hp={}'.format(alg, args.dist, d, s, l, hp))
            print("Failed")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""
    Simulations of "Better than Optimal: Improving Adaptive Stochastic Quantization Using Shared Randomness" paper.
    Running examples:
    $ python simba_exact_speed_error_tests.py --numseeds 10 --dist normal --verbose 

    see more options by running: $ python simba_exact_speed_error_tests.py --help
    """,
    formatter_class=argparse.RawTextHelpFormatter)

    ### verbosity
    parser.add_argument('--verbose', default=False, action='store_true', help='print detailed progress')

    ### seed
    parser.add_argument('--numseeds', default=10, type=int, help='Number of random seeds per sumulation')

    ### specific distribution
    parser.add_argument('--dist', default='lognormal', choices=['lognormal', 'normal', 'exponential', 'truncnorm', 'weibull'], help='specific distribution to run')
    
    ### specific m_quiver
    parser.add_argument('--m_quiver', default=1000, type=int, help='specific m_quiver')

    ### specific nbits
    parser.add_argument('--nbits', default=4, type=int, help='specific nbits')

    ### specific q
    parser.add_argument('--l', default=2, type=int, help='specific l') # TODO: add limit to l?

    args = parser.parse_args()

    if not os.path.isdir('results'):
        os.mkdir('results')

    seeds = [42, 104, 78, 45, 23, 38, 62, 101, 235, 1001]

    # default hyper parameters
    alg = 'simba_exact'
    iters = 1000000
    bin_iters = 2
    bin_iters_increase_threshold = .99
    stopping_threshold = .999
    debug = False
    log_cost=False 
    hp = (args.m_quiver, iters, bin_iters, bin_iters_increase_threshold, stopping_threshold)

    # simulation parameters
    vec_log2_dims = [10, 12, 14, 16, 18, 20, 22, 24]
    s = 2**args.nbits
    
    ### device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    # Simba's results dict structure is: results[METRIC][alg][args.dist][(vec_dim, nbits, m, q, iters, bin_iters, bin_iters_increase_threshold, stopping_threshold)
    
    results = {}
    # TODO: depreacate or modify
    results['alg'] = alg
    results['num_seeds'] = args.numseeds
    results['dist'] = args.dist
    results['vec_d_list'] = [2**x for x in vec_log2_dims]
    results['s'] = 2**args.nbits
    results['l'] = args.l
    results['m_quiver'] = args.m_quiver
    results['hp'] = hp

    results['sort_time[ms]'] = {}
    results['sqv_time[ms]'] = {}
    results['quantize_time[ms]'] = {}
    
    results['nmse'] = {}
    results['vec_norm'] = {}
    
    global_start = datetime.now()

    for seed_idx in range(args.numseeds):
        
        seed_start = datetime.now()
        seed = seeds[seed_idx]
        torch.cuda.empty_cache()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        print('******* seed {} (#{}) / {} distribution *******'.format(seed, seed_idx+1, args.dist))

        sampled_vecs = {}
        d_max = 2**vec_log2_dims[-1]

        if args.dist == 'lognormal':
            sampled_vecs[args.dist] = torch.distributions.LogNormal(0, 1).sample([d_max]).view(-1).to(device)
        elif args.dist == 'normal':
            sampled_vecs[args.dist] = torch.distributions.Normal(0, 1).sample([d_max]).view(-1).to(device)
        elif args.dist == 'exponential':
            sampled_vecs[args.dist] = torch.distributions.exponential.Exponential(torch.Tensor([1])).sample([d_max]).view(-1).to(device)
        elif args.dist == 'truncnorm':
            sampled_vecs[args.dist] = torch.tensor(truncated_normal([d_max]), device=device)
        elif args.dist == 'weibull':
            sampled_vecs[args.dist] = torch.distributions.weibull.Weibull(torch.tensor([1.0]),torch.tensor([1.0])).sample([d_max]).view(-1).to(device)

                    
        if alg not in results['sort_time[ms]'].keys():
            results['sort_time[ms]'][alg] = {}
            results['sqv_time[ms]'][alg] = {}
            results['quantize_time[ms]'][alg] = {}
            results['nmse'][alg] = {}
            results['vec_norm'][alg] = {}
                
        if args.dist not in results['sort_time[ms]'][alg].keys():
            results['sort_time[ms]'][alg][args.dist] = {}
            results['sqv_time[ms]'][alg][args.dist] = {}
            results['quantize_time[ms]'][alg][args.dist] = {}
            results['nmse'][alg][args.dist] = {}
            results['vec_norm'][alg][args.dist] = {}
            
        for vec_dim in vec_log2_dims:

            ### params
            d = 2**vec_dim

            vec = sampled_vecs[args.dist][:2 ** vec_dim].clone()
            if vec.numel() != 2 ** vec_dim:
                raise RuntimeError('vec length is not equal {} (2**{})'.format(2 ** vec_dim, vec_dim))

            if d not in results['sort_time[ms]'][alg][args.dist].keys():
                results['sort_time[ms]'][alg][args.dist][d] = {}
                results['sqv_time[ms]'][alg][args.dist][d] = {}
                results['quantize_time[ms]'][alg][args.dist][d] = {}
                results['nmse'][alg][args.dist][d] = {}
                results['vec_norm'][alg][args.dist][d] = []
            
            results['vec_norm'][alg][args.dist][d].append(torch.norm(vec, 2)**2)
                                        

            if s not in results['sort_time[ms]'][alg][args.dist][d].keys():
                results['sort_time[ms]'][alg][args.dist][d][s] = {}
                results['sqv_time[ms]'][alg][args.dist][d][s] = {}
                results['quantize_time[ms]'][alg][args.dist][d][s] = {}
                results['nmse'][alg][args.dist][d][s] = {}

            if args.l not in results['sort_time[ms]'][alg][args.dist][d][s].keys():
                results['sort_time[ms]'][alg][args.dist][d][s][args.l] = {}
                results['sqv_time[ms]'][alg][args.dist][d][s][args.l] = {}
                results['quantize_time[ms]'][alg][args.dist][d][s][args.l] = {}
                results['nmse'][alg][args.dist][d][s][args.l] = {}

            

            if seed_idx==0:
                results['sort_time[ms]'][alg][args.dist][d][s][args.l][hp] = []
                results['sqv_time[ms]'][alg][args.dist][d][s][args.l][hp] = []
                results['quantize_time[ms]'][alg][args.dist][d][s][args.l][hp] = []
                results['nmse'][alg][args.dist][d][s][args.l][hp] = []

            eval(alg, vec, results, d, s, args.l, iters, bin_iters, bin_iters_increase_threshold, stopping_threshold, args.m_quiver, log_cost, debug)

        print('****** End of round {}: {} seconds ******'.format(seed_idx+1, datetime.now()-seed_start))

    # save/update results    
    results_fn = './results/' + 'results_{}.pt'.format(args.dist)
    torch.save(results, results_fn)

    print('\nSimulations are finished (time[sec]={}).\nResults are saved to: {}'.format((datetime.now() - global_start), results_fn))
    