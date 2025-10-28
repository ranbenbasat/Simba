# Simba
Implementation of the Simba algorithm from the paper ``Better than Optimal: Improving Adaptive Stochastic Quantization Using Shared Randomness" (ACM SIGMETRICS 2026).

There are several ways to use the code:

* Use in Visual Studio by opening `Simba.sln`.

* Compile in Linux by running `make`.

* Install the Python package by running `python setup.py install` from `python_bindings/`.

We provide detailed usage examples in `python_bindings/tests/`.

We now give an example of how to reproduce Figure 9 (d) from the paper.
In this figure, we use `s=4` quantization values and compare Simba to QUIVER for the Normal(0,1) distribution. 

```python
import torch
from simba_eval import SimbaEvaluator
from simba_plots import SimbaPlotter

# Create evaluator
evaluator = SimbaEvaluator(device='cpu')
evaluator.verbose = True

# Run comprehensive evaluation
results = evaluator.run_comprehensive_evaluation(
    distributions=['normal'],
    sizes = [2**14, 2**16, 2**18, 2**20, 2**22, 2**24],
    s_values=[4],
    algorithms=['simba', 'quiver_exact'],
    algorithm_params={
        'simba': [{'l': 2}, {'l': 4}, {'l': 6}],
        'quiver_exact': [{}]
    }
)

filename = f'./results/figure_9d_data.pt'
evaluator.save_results(results, filename)

plotter = SimbaPlotter(filename)
plotter.plot_vnmse_vs_size('plots/figure_9d_vnmse_vs_size.png')
plotter.plot_time_vs_size( 'plots/figure_9d_time_vs_size.png')
```
