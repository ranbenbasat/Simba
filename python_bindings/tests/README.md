# Simba Quantization Algorithm Tests

This directory contains tests and evaluation tools for the Simba quantization algorithm, including comparisons with QUIVER algorithms.

## Quick Start

### Prerequisites
```bash
# Install Python dependencies
pip install torch numpy pandas matplotlib seaborn

# Build the C++ Python bindings
# cd /path/to/Simba/
cd python_bindings/
python setup.py install
```

### Running Tests

#### 1. Comprehensive Evaluation
```bash
cd tests/
python simba_eval.py
```
Runs a comprehensive evaluation across multiple algorithms, distributions, and parameters.

#### 2. Generate Plots
```bash
# Generate all plots from results file
python simba_plots.py --results ./results/simba_evaluation_normal_lognormal_exponential_uniform.pt

# Generate specific plot types
python simba_plots.py --results ./results/simba_evaluation_normal_lognormal_exponential_uniform.pt --plot vnmse_size

# Specify custom output directory
python simba_plots.py --results ./results/simba_evaluation_normal_lognormal_exponential_uniform.pt --output ./my_plots
```
Generates visualization plots from evaluation results.

## Test Files Overview

### Core Test Files

| File | Purpose | Description |
|------|---------|-------------|
| `simba_eval.py` | Comprehensive evaluation | Class-based evaluation framework for the algorithms |
| `simba_plots.py` | Visualization | Generates plots from evaluation results |

### Output Files

| File/Directory | Description |
|----------------|-------------|
| `results/` | Evaluation results (`.pt` files) |
| `plots/` | Generated visualization plots |

## Algorithm Hyperparameters

### Simba Algorithm

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `l` | int | 2 | Number of shared random values |
| `iters` | int | 100000 | Number of optimization iterations |
| `bin_iters` | int | 2 | Number of initial binary search iterations |
| `bin_iters_increase_threshold` | float | 0.99 | Relative MSE improvment threshold for increasing binary iterations |
| `stopping_threshold` | float | 0.999 | Relative MSE improvment convergence stopping threshold |
| `m_quiver_init` | int | 1000 | QUIVER parameter for Simba's initialization |
| `debug` | bool | False | Enable debug output |
| `log_cost_fn` | str | "" | Cost function logging filename (to plot time-to-vNMSE, this measures per-round cost) |

### QUIVER-Exact Algorithm

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `s` | int | Required | Number of quantization levels |

### QUIVER-Approx Algorithm

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `s` | int | Required | Number of quantization levels |
| `m` | int | 1000 | Grid resolution |

## Evaluation Framework

### SimbaEvaluator Class

The `SimbaEvaluator` class provides a comprehensive evaluation framework:

```python
import torch
from simba_eval import SimbaEvaluator

# Create evaluator
evaluator = SimbaEvaluator(device='cpu')

# Create test vector
d = 2**20
dist = torch.distributions.Normal(0, 1)
test_vector = dist.sample([d]).view(-1).double().sort()[0]

# Run single algorithm test
result = evaluator.evaluate_algorithm(
    algorithm='simba',
    vec=test_vector,
    s=4,
    l=2,
    iters=10000,
    m_quiver_init=1000
)

evaluator.verbose = True

# Run comprehensive evaluation
results = evaluator.run_comprehensive_evaluation(
    distributions=['normal', 'lognormal', 'uniform', 'exponential'],
    sizes = [2**16, 2**18],
    s_values=[2, 4, 8, 16, 32],
    algorithm_params={
        'simba': [{'l': 2}, {'l': 4}, {'l': 6}],
        'quiver_approx': [{'m': 100}, {'m': 1000}, {'m': 10000}],
        'quiver_exact': [{}]
    }
)

evaluator.print_summary(results)
```

### Example Distributions

- **Normal**: `torch.randn(size)` - Standard normal distribution
- **Lognormal**: `torch.distributions.LogNormal(0, 1).sample([size])` - Log-normal distribution
- **Uniform**: `torch.rand(size) * 2 - 1` - Uniform distribution on [-1, 1]
- **Exponential**: `torch.distributions.Exponential(1.0).sample([size])` - Exponential distribution

## Plotting System

### SimbaPlotter Class

The `SimbaPlotter` class generates various visualization plots:

```python
from simba_plots import SimbaPlotter

# Create plotter
plotter = SimbaPlotter('./results/simba_evaluation_normal_lognormal_exponential_uniform.pt')

# Generate all plots
plotter.generate_all_plots()

# Generate specific plots
plotter.plot_vnmse_vs_size('plots/vnmse_vs_size.png')
plotter.plot_time_vs_size('plots/time_vs_size.png')
plotter.plot_vnmse_vs_s('plots/vnmse_vs_s.png')
plotter.plot_time_vs_s('plots/time_vs_s.png')
```

### Command-Line Options

The `simba_plots.py` script supports several command-line arguments:

| Option | Default | Description |
|--------|---------|-------------|
| `--results` | `./results/simba_evaluation.pt` | Path to results file |
| `--output` | `./plots` | Output directory for plots |
| `--plot` | `all` | Which plot to generate (`all`, `vnmse_size`, `time_size`, `vnmse_s`, `comparison`, `summary`) |
| `--show` | False | Show plots instead of saving |

### Generated Plot Types

1. **vNMSE vs Vector Size**: Quantization quality vs input size
2. **Time vs Vector Size**: Quantization time vs input size  
3. **vNMSE vs s**: Quantization quality vs number of levels
4. **Time vs s**: Quantization time vs number of levels

### Algorithm Styling

Consistent styling across all plots:

| Algorithm | Marker | Line Style | Color |
|-----------|--------|------------|-------|
| Simba (l=2) | ○ | - | Blue |
| Simba (l=4) | □ | - | Orange |
| Simba (l=6) | ◆ | - | Green |
| QUIVER-Exact | △ | -- | Red |
| QUIVER-Approx (m=100) | ✕ | -. | Brown |
| QUIVER-Approx (m=1000) | ▽ | -. | Purple |
| QUIVER-Approx (m=10000) | + | : | Pink |

## Performance Metrics

### vNMSE (Variance-Normalized Mean Squared Error)

The primary quality metric used throughout the evaluation:

- **ASQ vNMSE**: For 1D quantization levels (QUIVER-Exact, QUIVER-Approx, no shared randomness)
- **AUQ vNMSE**: For 2D quantization levels (Simba, using shared randomness)

Formula: `vNMSE = ||X - Q(X)||² / ||X||²`

### Timing Measurements

- **Q Calculation Time**: Time to compute quantization levels

## File Structure

```
tests/
├── README.md                           # This file
├── simba_eval.py                       # Comprehensive evaluation framework
├── simba_plots.py                      # Plotting system
├── results/                            # Evaluation results
│   └── simba_evaluation_normal_lognormal_exponential_uniform.pt
└── plots/                             # Generated plots
    ├── normal/
    ├── lognormal/
    ├── uniform/
    └── exponential/
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure to build the C++ bindings first
   ```bash
   python setup.py install
   ```

2. **CUDA Error**: Set device of the Evaluator to 'cpu' if CUDA is not available (Simba runs on the CPU either way)
   ```python
   evaluator = SimbaEvaluator(device='cpu')
   ```

3. **Memory Error**: Reduce vector size or use smaller parameter sets
   ```python
   # Use smaller test vector
   test_vector = torch.randn(10000)  # Instead of 100000000
   ```

4. **File Not Found**: Ensure results files exist before plotting
   ```bash
   # Run evaluation first
   python simba_eval.py
   # Then generate plots (specify correct results file path)
   python simba_plots.py --results ./results/simba_evaluation_normal_lognormal_exponential_uniform.pt
   ```

### Performance Tips

- Use `device='cpu'` for more consistent results across systems
- Start with smaller parameter sets for initial testing
- Use `debug=True` to get a more detailed output during Simba execution
- Monitor memory usage with large vectors and high `l` values

