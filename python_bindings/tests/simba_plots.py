#!/usr/bin/env python3
"""
Simba evaluation results plotting script
Reads './results/simba_evaluation.pt' and generates comprehensive plots
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Any, Optional

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Define consistent styling for algorithms
ALGORITHM_STYLES = {
    'Simba (l=2)': {'marker': '.', 'linestyle': '--', 'linewidth': 2, 'markersize': 8, 'color': 'b'},
    'Simba (l=4)': {'marker': '.', 'linestyle': '--', 'linewidth': 2, 'markersize': 8, 'color': 'tab:purple'},
    'Simba (l=6)': {'marker': '.', 'linestyle': '--', 'linewidth': 2, 'markersize': 8, 'color': 'tab:pink'},
    'QUIVER-Exact': {'marker': '>', 'linestyle': '-', 'linewidth': 2, 'markersize': 8, 'color': 'red'},
    'QUIVER-Approx (m=100)': {'marker': 'x', 'linestyle': '-.', 'linewidth': 2, 'markersize': 8, 'color': '#8c564b'},
    'QUIVER-Approx (m=1000)': {'marker': '^', 'linestyle': '-.', 'linewidth': 2, 'markersize': 8, 'color': 'tab:cyan'},
    'QUIVER-Approx (m=10000)': {'marker': '+', 'linestyle': ':', 'linewidth': 2, 'markersize': 8, 'color': '#e377c2'},    
}

class SimbaPlotter:
    """
    Class for plotting Simba evaluation results
    """
    
    def __init__(self, results_file: str = './results/simba_evaluation.pt'):
        """
        Initialize the plotter with results file
        
        Args:
            results_file: Path to the results file
        """
        self.results_file = results_file
        self.results = None
        self.df = None
        self.distributions = None
        
    def load_results(self) -> Dict[str, Any]:
        """Load results from file"""
        if not Path(self.results_file).exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        self.results = torch.load(self.results_file, weights_only=False)
        print(f"Loaded results from: {self.results_file}")
        print(f"Total experiments: {len(self.results['results'])}")
        
        # Extract unique distributions
        if 'evaluation_config' in self.results and 'distributions' in self.results['evaluation_config']:
            self.distributions = self.results['evaluation_config']['distributions']
        else:
            # Fallback: extract from data
            df_temp = pd.DataFrame(self.results['results'])
            self.distributions = df_temp['distribution'].unique().tolist()
        
        print(f"Distributions found: {self.distributions}")
        return self.results
    
    def results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for easier plotting"""
        if self.results is None:
            self.load_results()
        
        # Extract successful results only
        successful_results = [r for r in self.results['results'] if r['success']]
        
        # Convert to DataFrame
        self.df = pd.DataFrame(successful_results)
        
        # Add derived columns
        self.df['log_size'] = np.log2(self.df['size'])
        self.df['log_s'] = np.log2(self.df['s'])
        
        # Create algorithm parameter string for better labeling
        def create_alg_label(row):
            alg = row['algorithm']
            if alg == 'simba':
                return f"Simba (l={int(row.get('l', 1))})"
            elif alg == 'quiver_approx':
                return f"QUIVER-Approx (m={int(row.get('m', 1000))})"
            elif alg == 'quiver_exact':
                return "QUIVER-Exact"
            else:
                return alg
        
        self.df['algorithm_label'] = self.df.apply(create_alg_label, axis=1)
        
        print(f"DataFrame created with {len(self.df)} successful experiments")
        return self.df
    
    def get_data_for_distribution(self, distribution: str) -> pd.DataFrame:
        """Get data filtered for a specific distribution"""
        if self.df is None:
            self.results_to_dataframe()
        
        dist_data = self.df[self.df['distribution'] == distribution].copy()
        print(f"Data for distribution '{distribution}': {len(dist_data)} experiments")
        return dist_data
    
    def plot_vnmse_vs_size(self, save_path: Optional[str] = None, distribution: Optional[str] = None, s: Optional[int] = None):
        """Plot vNMSE vs vector size for different algorithms"""
        if self.df is None:
            self.results_to_dataframe()
        
        # Filter data by distribution if specified
        if distribution is not None:
            plot_data = self.get_data_for_distribution(distribution)
        else:
            plot_data = self.df
        
        # Filter data by s value if specified, otherwise use the largest available
        if s is None:
            s = int(plot_data['s'].max())
        
        plot_data = plot_data[plot_data['s'] == s].copy()
        print(f"Using s={s} for vnmse_vs_size plot: {len(plot_data)} experiments")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by algorithm and size, calculate mean vNMSE
        for alg_label in plot_data['algorithm_label'].unique():
            alg_data = plot_data[plot_data['algorithm_label'] == alg_label]
            
            # Calculate mean and std vNMSE for each size
            size_stats = alg_data.groupby('size').agg({
                'asq_vnmse': ['mean', 'std'],
                'auq_vnmse': ['mean', 'std']
            }).reset_index()
            
            # Flatten column names
            size_stats.columns = ['size', 'asq_mean', 'asq_std', 'auq_mean', 'auq_std']
            
            # Use ASQ vNMSE if available, otherwise AUQ vNMSE
            vnmse_values = []
            vnmse_errors = []
            for _, row in size_stats.iterrows():
                if pd.notna(row['asq_mean']):
                    vnmse_values.append(row['asq_mean'])
                    vnmse_errors.append(row['asq_std'] if pd.notna(row['asq_std']) else 0)
                elif pd.notna(row['auq_mean']):
                    vnmse_values.append(row['auq_mean'])
                    vnmse_errors.append(row['auq_std'] if pd.notna(row['auq_std']) else 0)
                else:
                    vnmse_values.append(np.nan)
                    vnmse_errors.append(0)
            
            # Get styling for this algorithm
            style = ALGORITHM_STYLES.get(alg_label, {'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'markersize': 8})
            
            ax.errorbar(
                size_stats['size'],
                vnmse_values,
                vnmse_errors,
                label=alg_label,
                capsize=8,
                capthick=1.5,
                **style
            )
        
        ax.set_xlabel('Vector Size', fontsize=12)
        ax.set_ylabel('vNMSE', fontsize=12)
        ax.set_title(f'vNMSE vs Vector Size (s={s})', fontsize=14, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:   
            plt.show()
    
    def plot_time_vs_size(self, save_path: Optional[str] = None, distribution: Optional[str] = None):
        """Plot quantization time vs vector size"""
        if self.df is None:
            self.results_to_dataframe()
        
        # Filter data by distribution if specified
        if distribution is not None:
            plot_data = self.get_data_for_distribution(distribution)
        else:
            plot_data = self.df
        
        fig, ax = plt.subplots(figsize=(12, 8))
        s = plot_data['s'].max()
        
        for alg_label in plot_data['algorithm_label'].unique():
            alg_data = plot_data[plot_data['algorithm_label'] == alg_label]
            alg_data = alg_data[(alg_data['s'] == s)]
            
            # Calculate mean and std time for each size
            size_stats = alg_data.groupby('size')['quantization_time_ms'].agg(['mean', 'std']).reset_index()
            
            # Get styling for this algorithm
            style = ALGORITHM_STYLES.get(alg_label, {'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'markersize': 8})
            
            ax.errorbar(
                size_stats['size'],
                size_stats['mean'],
                size_stats['std'],
                label=alg_label,
                capsize=8,
                capthick=1.5,
                **style
            )
        
        ax.set_xlabel('Vector Size', fontsize=12)
        ax.set_ylabel('Quantization Time (ms)', fontsize=12)
        ax.set_title(f'Quantization Time vs Vector Size (s={s})', fontsize=14, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_vnmse_vs_s(self, save_path: Optional[str] = None, distribution: Optional[str] = None):
        """Plot vNMSE vs number of quantization levels (s)"""
        if self.df is None:
            self.results_to_dataframe()
        
        # Filter data by distribution if specified
        if distribution is not None:
            plot_data = self.get_data_for_distribution(distribution)
        else:
            plot_data = self.df
        
        fig, ax = plt.subplots(figsize=(12, 8))
        d = plot_data['input_size'].max()
        
        for alg_label in plot_data['algorithm_label'].unique():
            alg_data = plot_data[(plot_data['algorithm_label'] == alg_label)]
            alg_data = alg_data[(alg_data['input_size'] == d)]
            
            
            # Calculate mean and std vNMSE for each s
            s_stats = alg_data.groupby('s').agg({
                'asq_vnmse': ['mean', 'std'],
                'auq_vnmse': ['mean', 'std']
            }).reset_index()
            

            # Flatten column names
            s_stats.columns = ['s', 'asq_mean', 'asq_std', 'auq_mean', 'auq_std']
            
            # Use ASQ vNMSE if available, otherwise AUQ vNMSE
            vnmse_values = []
            vnmse_errors = []
            for _, row in s_stats.iterrows():
                if pd.notna(row['asq_mean']):
                    vnmse_values.append(row['asq_mean'])
                    vnmse_errors.append(row['asq_std'] if pd.notna(row['asq_std']) else 0)
                elif pd.notna(row['auq_mean']):
                    vnmse_values.append(row['auq_mean'])
                    vnmse_errors.append(row['auq_std'] if pd.notna(row['auq_std']) else 0)
                else:
                    vnmse_values.append(np.nan)
                    vnmse_errors.append(0)
            
            # Get styling for this algorithm
            style = ALGORITHM_STYLES.get(alg_label, {'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'markersize': 8})
            
            ax.errorbar(
                s_stats['s'],
                vnmse_values,
                vnmse_errors,
                label=alg_label,
                capsize=8,
                capthick=1.5,
                **style
            )
        
        ax.set_xlabel('Number of Quantization Levels (s)', fontsize=12)
        ax.set_ylabel('vNMSE', fontsize=12)
        ax.set_title('vNMSE vs Number of Quantization Levels ($d=' + str(d) + '$)', fontsize=14, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_time_vs_s(self, save_path: Optional[str] = None, distribution: Optional[str] = None):
        """Plot quantization time vs number of quantization levels (s) for different algorithms"""
        if self.df is None:
            self.results_to_dataframe()
        
        # Filter data by distribution if specified
        if distribution is not None:
            plot_data = self.get_data_for_distribution(distribution)
        else:
            plot_data = self.df
        
        fig, ax = plt.subplots(figsize=(12, 8))
        d = plot_data['input_size'].max()
        
        for alg_label in plot_data['algorithm_label'].unique():
            alg_data = plot_data[plot_data['algorithm_label'] == alg_label]
            alg_data = alg_data[(alg_data['input_size'] == d)]
            
            # Calculate mean time for each s value
            s_stats = alg_data.groupby('s')['quantization_time_ms'].agg(['mean', 'std']).reset_index()
            
            # Get styling for this algorithm
            style = ALGORITHM_STYLES.get(alg_label, {'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'markersize': 8})
            
            ax.errorbar(
                s_stats['s'],
                s_stats['mean'],
                s_stats['std'],
                label=alg_label,
                capsize=8,
                capthick=1.5,
                **style
            )
        
        ax.set_xlabel('Number of Quantization Levels (s)')
        ax.set_ylabel('Quantization Time (ms)')
        ax.set_title('Quantization Time vs Number of Quantization Levels ($d=' + str(d) + '$)')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    
    def print_summary_stats(self):
        """Print summary statistics"""
        if self.df is None:
            self.results_to_dataframe()
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY STATISTICS")
        print("="*60)
        
        # Overall stats
        total_experiments = len(self.results['results'])
        successful_experiments = len(self.df)
        failed_experiments = total_experiments - successful_experiments
        
        print(f"Total experiments: {total_experiments}")
        print(f"Successful: {successful_experiments}")
        print(f"Failed: {failed_experiments}")
        print(f"Success rate: {successful_experiments/total_experiments*100:.1f}%")
        
        # Algorithm-specific stats
        print(f"\nAlgorithm Performance:")
        for alg_label in self.df['algorithm_label'].unique():
            alg_data = self.df[self.df['algorithm_label'] == alg_label]
            
            # Get vNMSE values
            vnmse_values = []
            for _, row in alg_data.iterrows():
                asq_val = row.get('asq_vnmse')
                auq_val = row.get('auq_vnmse')
                
                if pd.notna(asq_val):
                    vnmse_values.append(asq_val)
                elif pd.notna(auq_val):
                    vnmse_values.append(auq_val)
            
            if vnmse_values:
                avg_vnmse = np.mean(vnmse_values)
                std_vnmse = np.std(vnmse_values)
                avg_time = np.mean(alg_data['quantization_time_ms'])
                
                print(f"  {alg_label}:")
                print(f"    Avg vNMSE: {avg_vnmse:.6f} ± {std_vnmse:.6f}")
                print(f"    Avg Time: {avg_time:.2f} ms")
                print(f"    Experiments: {len(alg_data)}")
    
    def generate_all_plots(self, output_dir: str = './plots'):
        """Generate all plots and save them to output directory"""
        if self.results is None:
            self.load_results()
        
        # Create plots for each distribution
        for distribution in self.distributions:
            dist_output_dir = f"{output_dir}/{distribution}"
            Path(dist_output_dir).mkdir(parents=True, exist_ok=True)
            
            print(f"Generating plots for distribution '{distribution}' in: {dist_output_dir}")
            
            # Individual plots for this distribution
            self.plot_vnmse_vs_size(f"{dist_output_dir}/vnmse_vs_size.png", distribution)
            self.plot_time_vs_size(f"{dist_output_dir}/time_vs_size.png", distribution)
            self.plot_vnmse_vs_s(f"{dist_output_dir}/vnmse_vs_s.png", distribution)
            self.plot_time_vs_s(f"{dist_output_dir}/time_vs_s.png", distribution)
            
            print(f"Plots for '{distribution}' saved to: {dist_output_dir}")
        
        print(f"All plots saved to: {output_dir}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Plot Simba evaluation results')
    parser.add_argument('--results', default='./results/simba_evaluation_normal_lognormal_exponential_uniform.pt',
                       help='Path to results file')
    parser.add_argument('--output', default='./plots',
                       help='Output directory for plots')
    parser.add_argument('--plot', choices=['all', 'vnmse_size', 'time_size', 'vnmse_s', 
                                         'comparison', 'summary'],
                       default='all', help='Which plot to generate')
    parser.add_argument('--show', action='store_true',
                       help='Show plots instead of saving')
    
    args = parser.parse_args()
    
    # Create plotter
    plotter = SimbaPlotter(args.results)
    
    try:
        # Load results
        plotter.load_results()
        plotter.results_to_dataframe()
        
        # Print summary
        plotter.print_summary_stats()
        
        # Generate plots
        if args.plot == 'all':
            if args.show:
                plotter.plot_vnmse_vs_size()
                plotter.plot_time_vs_size()
                plotter.plot_vnmse_vs_s()
                plotter.plot_algorithm_comparison()
                plotter.plot_heatmap('asq_vnmse')
            else:
                plotter.generate_all_plots(args.output)
        elif args.plot == 'vnmse_size':
            plotter.plot_vnmse_vs_size(None if args.show else f"{args.output}/vnmse_vs_size.png")
        elif args.plot == 'time_size':
            plotter.plot_time_vs_size(None if args.show else f"{args.output}/time_vs_size.png")
        elif args.plot == 'vnmse_s':
            plotter.plot_vnmse_vs_s(None if args.show else f"{args.output}/vnmse_vs_s.png")
        elif args.plot == 'comparison':
            plotter.plot_algorithm_comparison(None if args.show else f"{args.output}/algorithm_comparison.png")
        elif args.plot == 'summary':
            pass  # Summary already printed above
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run the evaluation first to generate results.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
