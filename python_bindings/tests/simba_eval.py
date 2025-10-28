#!/usr/bin/env python3

import torch
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import os

# Import the simba_cpp module
import simba_cpp

class SimbaEvaluator:
    """
    A class for evaluating Simba and comparing it to QUIVER.
    """
    
    def __init__(self, device: str = 'auto', verbose: bool = False):
        """
        Initialize the Simba and QUIVER evaluator.
        
        Args:
            device: Device to use ('auto', 'cpu', 'cuda')
            verbose: Whether to print detailed progress
        """
        self.device = self._setup_device(device)
        self.verbose = verbose
        self.results = {}
        
    def _setup_device(self, device: str) -> str:
        """Setup the computation device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def generate_test_data(self, size: int, distribution: str = 'normal', 
                          seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate test data for evaluation.
        
        Args:
            size: Size of the vector
            distribution: Type of distribution ('normal', 'lognormal', 'exponential', 'uniform')
            seed: Random seed for reproducibility
            
        Returns:
            Generated test vector
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        if distribution == 'normal':
            data = torch.randn(size, device=self.device, dtype=torch.float64)
        elif distribution == 'lognormal':
            data = torch.distributions.LogNormal(0, 1).sample([size]).view(-1).to(self.device)
        elif distribution == 'exponential':
            data = torch.distributions.Exponential(1.0).sample([size]).view(-1).to(self.device)
        elif distribution == 'uniform':
            data = torch.rand(size, device=self.device, dtype=torch.float64) * 2 - 1
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
            
        return data.double()
    
    def quantize_vector(self, vec: torch.Tensor, sqv: torch.Tensor) -> torch.Tensor:
        """
        Quantize a vector using the given quantization levels.
        
        Args:
            vec: Input vector to quantize
            sqv: Quantization levels (1D or 2D tensor)
            
        Returns:
            Quantized vector
        """
        if len(sqv.size()) == 1:
            # Simple quantization (no shared randomness)
            buckets = torch.bucketize(vec, sqv)
            up = torch.take(sqv, buckets)
            down = torch.take(sqv, torch.clamp(buckets - 1, min=0))
            
            p = (up - vec) / (up - down)
            r = torch.rand(p.numel(), device=self.device)
            
            return down + (up - down) * (p < r)
        else:
            # Simba's quantization (with shared randomness)
            return self._quantize_with_shared_randomness(vec, sqv)
    
    def _quantize_with_shared_randomness(self, vec: torch.Tensor, sqv: torch.Tensor) -> torch.Tensor:
        """Quantize using shared randomness."""
        thresholds = self._calc_thresholds(sqv)
        buckets = torch.bucketize(vec, thresholds)
        
        q, s = sqv.size()
        H = buckets % q
        S = torch.div(buckets, q, rounding_mode='trunc')
        d = vec.numel()
        SR = torch.randint(q, (d,), device=self.device)
        
        larger_than_SR = SR < H
        smaller_than_SR = SR > H
        equal_to_SR = SR == H
        nr_equal_to_SR = torch.sum(equal_to_SR)
        
        det_sum = torch.zeros(nr_equal_to_SR, device=self.device).double()
        Sequal_to_SR = S[equal_to_SR]
        Hequal_to_SR = H[equal_to_SR]
        
        for H_prime in range(q):
            larger_than_H_prime = H_prime < Hequal_to_SR
            smaller_than_H_prime = H_prime > Hequal_to_SR
            det_sum[larger_than_H_prime] += sqv[H_prime][Sequal_to_SR[larger_than_H_prime]+1]
            det_sum[smaller_than_H_prime] += sqv[H_prime][Sequal_to_SR[smaller_than_H_prime]]
        
        x_p = q * vec[equal_to_SR] - det_sum
        res = torch.zeros(d, device=self.device, dtype=torch.double)
        res[larger_than_SR] = sqv[SR[larger_than_SR], S[larger_than_SR]+1]
        res[smaller_than_SR] = sqv[SR[smaller_than_SR], S[smaller_than_SR]]
        
        sqv_equal_to_SR_S = sqv[H, S][equal_to_SR]
        S_plus_1 = torch.clamp(S+1, max=sqv.size()[1]-1)
        sqv_equal_to_SR_S_plus_1 = sqv[H, S_plus_1][equal_to_SR]
        sqv_equal_to_SR_S_plus_1___minus___sqv_equal_to_SR_S = sqv_equal_to_SR_S_plus_1 - sqv_equal_to_SR_S
        p_up = (x_p - sqv_equal_to_SR_S) / sqv_equal_to_SR_S_plus_1___minus___sqv_equal_to_SR_S
        r = torch.rand(nr_equal_to_SR, device=self.device)
        res[equal_to_SR] = sqv_equal_to_SR_S + sqv_equal_to_SR_S_plus_1___minus___sqv_equal_to_SR_S * (r < p_up)
        
        return res
    
    def _calc_thresholds(self, sqv: torch.Tensor) -> torch.Tensor:
        """Calculate thresholds for Simba's quantization."""
        q, s = sqv.size()
        thresholds = torch.zeros(q*(s-1), device=self.device)
        T = sum([sqv[j][0] for j in range(1, q)]) + sqv[0][1]
        thresholds[0] = T
        idx = 1
        
        for H in range(1, q):
            T += sqv[H][1] - sqv[H][0]
            thresholds[idx] = T
            idx += 1
            
        for X in range(1, s-1):
            for H in range(0, q):
                T += sqv[H][X + 1] - sqv[H][X]
                thresholds[idx] = T
                idx += 1
                
        return thresholds / q
    
    def evaluate_algorithm(self, algorithm: str, vec: torch.Tensor, 
                          s: int, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a quantization algorithm.
        
        Args:
            algorithm: Algorithm name ('simba', 'quiver_exact', 'quiver_approx')
            vec: Input vector
            s: Number of quantization levels
            **kwargs: Additional algorithm-specific parameters
                     - For simba: l, iters, bin_iters, bin_iters_increase_threshold, stopping_threshold, m_quiver_init, debug, log_cost_fn
                     - For quiver_approx: m (approximation parameter)
            
        Returns:
            Dictionary with evaluation results
        """
        results = {
            'algorithm': algorithm,
            'input_size': vec.numel(),
            's': s,
            'success': False,
            'quantization_time_ms': None,
            'quantize_time_ms': None,
            'asq_vnmse': None,
            'auq_vnmse': None,
            'quantization_levels': None,
            'error': None
        }
        
        # Add algorithm-specific parameters to results
        if algorithm == 'simba':
            results['l'] = int(kwargs.get('l', 2))
            results['m'] = pd.NA
            results['m_quiver_init'] = int(kwargs.get('m_quiver_init', 1000))
            results['iters'] = int(kwargs.get('iters', 1000))
            results['bin_iters'] = int(kwargs.get('bin_iters', 2))
            results['bin_iters_increase_threshold'] = float(kwargs.get('bin_iters_increase_threshold', 0.99))
            results['stopping_threshold'] = float(kwargs.get('stopping_threshold', 0.999))
            results['debug'] = bool(kwargs.get('debug', False))
            results['log_cost_fn'] = str(kwargs.get('log_cost_fn', ""))
        elif algorithm == 'quiver_approx':
            results['l'] = pd.NA
            results['m'] = int(kwargs.get('m', 1000))
            results['m_quiver_init'] = pd.NA
            results['iters'] = pd.NA
            results['bin_iters'] = pd.NA
            results['bin_iters_increase_threshold'] = pd.NA
            results['stopping_threshold'] = pd.NA
            results['debug'] = pd.NA
            results['log_cost_fn'] = pd.NA
        else:  # quiver_exact
            results['l'] = pd.NA
            results['m'] = pd.NA
            results['m_quiver_init'] = pd.NA
            results['iters'] = pd.NA
            results['bin_iters'] = pd.NA
            results['bin_iters_increase_threshold'] = pd.NA
            results['stopping_threshold'] = pd.NA
            results['debug'] = pd.NA
            results['log_cost_fn'] = pd.NA 

        try:
            # Sort vector and move to CPU for C++ bindings
            svec, _ = torch.sort(vec)
            svec = svec.double().to('cpu')
            
            # Measure quantization time
            start_time = time.time_ns()
            
            if algorithm == 'simba':
                l = kwargs.get('l', 2)  
                sqv = simba_cpp.simba(
                    svec, s, l,
                    kwargs.get('iters', 1000),
                    kwargs.get('bin_iters', 2),
                    kwargs.get('bin_iters_increase_threshold', 0.99),
                    kwargs.get('stopping_threshold', 0.999),
                    kwargs.get('m_quiver_init', 1000),
                    kwargs.get('debug', False),
                    kwargs.get('log_cost_fn', "")
                )
            elif algorithm == 'quiver_approx':
                m = kwargs.get('m', 1000)
                sqv = simba_cpp.quiver_approx(svec, s, m)
            elif algorithm == 'quiver_exact':
                sqv = simba_cpp.quiver_exact_accelerated(svec, s)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            end_time = time.time_ns()
            results['quantization_time_ms'] = (end_time - start_time) / 1_000_000
            
            # Move results back to device
            sqv = sqv.to(self.device)
            results['quantization_levels'] = sqv.cpu().numpy()
            
            # Measure quantization time
            start_quantize = time.time_ns()
            quantized_vec = self.quantize_vector(vec, sqv)
            end_quantize = time.time_ns()
            results['quantize_time_ms'] = (end_quantize - start_quantize) / 1_000_000
            
            # Calculate vNMSE metrics
            # Note: vNMSE functions expect the sorted vector, not the original unsorted one
            sorted_vec_cpu = svec.double().to('cpu')
            if len(sqv.size()) == 1:
                # ASQ (Adaptive Stochastic Quantization)
                vnmse_val = simba_cpp.asq_vnmse(sorted_vec_cpu, sqv.to('cpu')).item()
                results['asq_vnmse'] = vnmse_val
            else:
                # AUQ (Adaptive Unbiased Quantization with Shared Randomness)
                vnmse_val = simba_cpp.auq_vnmse(sorted_vec_cpu, sqv.to('cpu')).item()
                results['auq_vnmse'] = vnmse_val
            
            results['success'] = True
            
            if self.verbose:
                param_str = f"s={s}"
                if algorithm == 'simba':
                    param_str += f", l={results['l']}"
                elif algorithm == 'quiver_approx':
                    param_str += f", m={results['m']}"
                
                # Get vNMSE value, handling NaN values
                asq_val = results.get('asq_vnmse')
                auq_val = results.get('auq_vnmse')
                
                if asq_val is not None and not (isinstance(asq_val, float) and np.isnan(asq_val)):
                    vnmse_val = asq_val
                elif auq_val is not None and not (isinstance(auq_val, float) and np.isnan(auq_val)):
                    vnmse_val = auq_val
                else:
                    vnmse_val = None
                
                vnmse_str = f"{vnmse_val:.6f}" if vnmse_val is not None else "N/A"
                print(f"✓ {algorithm}: {param_str}, vNMSE={vnmse_str}, "
                      f"time={results['quantization_time_ms']:.2f}ms")
            
        except Exception as e:
            results['error'] = str(e)
            if self.verbose:
                print(f"✗ {algorithm} failed: {e}")
        
        return results
    
    def run_comprehensive_evaluation(self, sizes: List[int], 
                                   distributions: List[str] = ['normal'],
                                   algorithms: List[str] = ['simba'],
                                   s_values: List[int] = [4, 8, 16],
                                   algorithm_params: Dict[str, List[Dict]] = None,
                                   num_seeds: int = 5) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across multiple parameters.
        
        Args:
            sizes: List of vector sizes to test
            distributions: List of distributions to test
            algorithms: List of algorithms to test
            s_values: List of quantization levels to test
            algorithm_params: Dict mapping algorithm names to lists of parameter dictionaries.
                             Each parameter dict can contain any valid parameters for that algorithm.
                             Example: {'simba': [{'l': 2}, {'l': 4, 'm_quiver_init': 1000}]}
            num_seeds: Number of random seeds per configuration
            
        Returns:
            Comprehensive evaluation results
        """
        if algorithm_params is None:
            algorithm_params = {
                'simba': [{'l': 2}, {'l': 4}, {'l': 6}],
                'quiver_approx': [{'m': 1000}, {'m': 10000}],
                'quiver_exact': [{}]
            }
        
        seeds = [42, 104, 78, 45, 23, 38, 62, 101, 235, 1001][:num_seeds]
        
        results = {
            'evaluation_config': {
                'sizes': sizes,
                'distributions': distributions,
                'algorithms': algorithms,
                's_values': s_values,
                'algorithm_params': algorithm_params,
                'num_seeds': num_seeds,
                'device': self.device
            },
            'results': []
        }
        
        # Calculate total experiments
        total_experiments = 0
        for alg in algorithms:
            param_count = len(algorithm_params.get(alg, [{}]))
            total_experiments += len(sizes) * len(distributions) * len(s_values) * param_count * num_seeds
        
        experiment_count = 0
        
        print(f"Starting comprehensive evaluation: {total_experiments} experiments")
        print(f"Device: {self.device}")
        
        for size in sizes:
            for dist in distributions:
                for alg in algorithms:
                    for s in s_values:
                        # Get algorithm-specific parameters
                        alg_params = algorithm_params.get(alg, [{}])
                        for param_dict in alg_params:
                            for seed in seeds:
                                experiment_count += 1
                                
                                if self.verbose:
                                    # Create parameter string for display
                                    param_str = ", ".join([f"{k}={v}" for k, v in param_dict.items()])
                                    if param_str:
                                        param_str = f", {param_str}"
                                    print(f"\n[{experiment_count}/{total_experiments}] "
                                          f"Size={size}, Dist={dist}, Alg={alg}, s={s}{param_str}, Seed={seed}")
                                
                                # Generate test data
                                vec = self.generate_test_data(size, dist, seed)
                                
                                # Run evaluation with all parameters from param_dict
                                result = self.evaluate_algorithm(alg, vec, s, **param_dict)
                                
                                # Add metadata
                                result.update({
                                    'size': size,
                                    'distribution': dist,
                                    'seed': seed,
                                    'experiment_id': experiment_count
                                })
                                
                                results['results'].append(result)
        
        print(f"\nEvaluation completed: {len(results['results'])} experiments")
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save evaluation results to file."""
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        torch.save(results, filename)
        print(f"Results saved to: {filename}")
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load evaluation results from file."""
        return torch.load(filename)
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of evaluation results."""
        successful_results = [r for r in results['results'] if r['success']]
        failed_results = [r for r in results['results'] if not r['success']]
        
        print(f"\n=== Evaluation Summary ===")
        print(f"Total experiments: {len(results['results'])}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        
        if successful_results:
            print(f"\n=== Performance Summary ===")
            algorithms = set(r['algorithm'] for r in successful_results)
            for alg in algorithms:
                alg_results = [r for r in successful_results if r['algorithm'] == alg]
                avg_time = np.mean([r['quantization_time_ms'] for r in alg_results if r['quantization_time_ms'] is not None])
                
                # Collect valid vNMSE values
                vnmse_values = []
                for r in alg_results:
                    asq_val = r.get('asq_vnmse')
                    auq_val = r.get('auq_vnmse')
                    
                    if asq_val is not None and not (isinstance(asq_val, float) and np.isnan(asq_val)):
                        vnmse_values.append(asq_val)
                    elif auq_val is not None and not (isinstance(auq_val, float) and np.isnan(auq_val)):
                        vnmse_values.append(auq_val)
                
                if vnmse_values:
                    avg_vnmse = np.mean(vnmse_values)
                    print(f"{alg}: avg_time={avg_time:.2f}ms, avg_vNMSE={avg_vnmse:.6f}")
                else:
                    print(f"{alg}: avg_time={avg_time:.2f}ms, avg_vNMSE=N/A")


def main():
    """Example usage of the SimbaEvaluator class."""
    evaluator = SimbaEvaluator(verbose=True)
    
    # Quick test
    print("Running quick evaluation...")
    vec = evaluator.generate_test_data(10000, 'normal', seed=42)
    result = evaluator.evaluate_algorithm('simba', vec, s=4, l=2)
    print(f"Quick test result: {result}")
    
    # Comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    results = evaluator.run_comprehensive_evaluation(
        sizes=[2**10, 2**12, 2**14, 2**16, 2**18],
        distributions=['normal', 'lognormal', 'exponential', 'uniform'],
        algorithms=['simba', 'quiver_exact', 'quiver_approx'],
        s_values=[2, 4, 8, 16, 32],
        algorithm_params={
            'simba': [
                {'l': 2, 'm_quiver_init': 1000, 'iters': 100000, 'bin_iters': 2, 
                 'bin_iters_increase_threshold': 0.99, 'stopping_threshold': 0.999},
                {'l': 4, 'm_quiver_init': 1000, 'iters': 100000, 'bin_iters': 2, 
                 'bin_iters_increase_threshold': 0.99, 'stopping_threshold': 0.999},
                {'l': 6, 'm_quiver_init': 1000, 'iters': 100000, 'bin_iters': 2, 
                 'bin_iters_increase_threshold': 0.99, 'stopping_threshold': 0.999}
            ],
            'quiver_approx': [{'m': 100}, {'m': 1000}, {'m': 10000}],
            'quiver_exact': [{}]
        },
        num_seeds=5
    )
    
    evaluator.print_summary(results)
    
    # Create filename with distribution
    distributions = results['evaluation_config']['distributions']
    dist_str = '_'.join(distributions)
    filename = f'./results/simba_evaluation_{dist_str}.pt'
    evaluator.save_results(results, filename)


if __name__ == '__main__':
    main()
