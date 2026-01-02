# STEADY: Stein-type Estimator Assisted by Dynamics

This repository contains the official Python implementation and experimental code for the paper:

**Stein-type Estimator Assisted by Dynamics**  
*Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26), August 09--13, 2026, Jeju Island, Republic of Korea*

**DOI:** [10.1145/3770854.3780166](https://doi.org/10.1145/3770854.3780166)  
**Source Code:** [https://github.com/shanfenghu/steady](https://github.com/shanfenghu/steady)

## Abstract

Estimating the equilibrium parameters of environmental systems is a fundamental task, yet it is often hampered by sparse and noisy sensor data. While the standard Maximum Likelihood Estimator (MLE) is intuitive, Stein's paradox famously shows that it is statistically inefficient in high dimensions. To address this, we introduce *STEADY*, an estimator that generalises Stein's paradox by integrating physical knowledge. We derive our estimator from a principled empirical Bayes model where the prior distribution over the equilibria is a direct consequence of the stationary properties of the system's governing differential equations. This leads to a novel adaptive shrinkage mechanism, where the amount of shrinkage applied to each observation is naturally modulated by the physical stability of the measured system. We provide a rigorous frequentist analysis of our estimator, proving that STEADY not only dominates the MLE but is also minimax under certain conditions, offering the strongest possible guarantee of robustness. We validate our claims on synthetic data and demonstrate STEADY's utility on the global Argo ocean float dataset, showing that it effectively filters noise to reveal the "North Atlantic Warming Hole".

## Quick Usage

The `steady` package is simple to use. Here is a minimal example of applying the STEADY estimator to a small, synthetic dataset.

```python
import numpy as np
from steady.estimators import steady_estimator

# 1. Define the problem
Y = np.array([0.5, 5.2, 9.8])  # Noisy observations
lambdas = np.array([1.0, 0.5, 0.2]) # Physical stability for each observation
mu_phys = 4.0 # The global physical prior
sigma_ou_sq = 2.0 # Physical process variance
sigma_obs_sq = 1.0 # Measurement variance

# 2. Get the STEADY estimate
mu_hat_steady = steady_estimator(Y, lambdas, mu_phys, sigma_ou_sq, sigma_obs_sq)

print(f"Original Observations (MLE): {Y}")
print(f"STEADY Estimates: {np.round(mu_hat_steady, 2)}")
# Expected Output:
# Original Observations (MLE): [0.5 5.2 9.8]
# STEADY Estimates: [1. 4.96  7.88]
```

## Installation

This project uses Python 3.11. To set up the environment and install the necessary dependencies, follow these steps:

1.  Clone the repository:
    ```bash
    git clone https://github.com/shanfenghu/steady.git
    cd steady
    ```

2.  **Create and activate a conda environment:**
    ```bash
    conda create -n steady_env python=3.11
    conda activate steady_env
    ```

3.  **Install the package and dependencies:**
    The project is packaged with `pyproject.toml`. Install it in editable mode, which will also install all required dependencies (`numpy`, `matplotlib`, `cartopy`, etc.).
    ```bash
    pip install -e .
    ```

## Reproducing Experimental Results

All experimental results can be reproduced by running the scripts in the `experiments/` directory.

### 1. Data Setup

Before running the real-world experiment, you must download the Argo ocean data. Please follow the instructions in the `data/README.md` file to download the required NetCDF file and place it in the `data/` directory.

### 2. Running the Experiments

All experiment scripts can be run from the root directory of the project.

* **Generate the Data Processing Visualization:**
    ```bash
    python experiments/process_argo.py --data_path data/your_argo_file.nc
    ```
    This will also create the `argo_processed_data.csv` file required for the main Argo experiment.

* **Generate Synthetic Experiment Results:**
    ```bash
    # Dominance over MLE experiment
    python experiments/exp_dominance.py

    # Model Correctness Condition experiment
    python experiments/exp_correctness.py

    # Robustness to Parameter Misspecification experiment
    python experiments/exp_sensitivity.py

    # Computational Scalability experiment
    python experiments/exp_scalability.py
    ```

* **Generate Real-World Case Study Results:**
    This script uses the processed data file created by `process_argo.py`.
    ```bash
    python experiments/exp_argo.py --data-path data/argo_processed_data.csv
    ```

All figures will be saved to the `figures/` directory, and the quantitative results table will be saved to the `results/` directory.

## Code Structure

The project is organized into a clean, reproducible structure:

* `steady/`: The core, installable Python package.
    * `estimators.py`: Contains the implementations of the STEADY, GJS, and MLE estimators.
    * `simulation.py`: Contains the function for simulating the Ornstein-Uhlenbeck process.
* `experiments/`: Scripts to reproduce all experimental results.
    * `process_argo.py`: Preprocesses the raw Argo data and generates the diagnostic visualization.
    * `exp_argo.py`: Runs the main Argo case study and generates the results maps and quantitative table.
    * `exp*.py`: Scripts for the four synthetic experiments.
    * `plot_utils.py`: Shared plotting styles and functions for consistent figures.
* `tests/`: Unit tests for the core library functions.
* `data/`: Directory for storing the raw and processed data.
* `pyproject.toml`: The package definition file.

## Citation

If you use STEADY in your research, please cite:

```bibtex
@inproceedings{hu2026steady,
  title={Stein-type Estimator Assisted by Dynamics},
  author={Hu, Shanfeng and Aslam, Nauman},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year={2026},
  pages={},
  doi={10.1145/3770854.3780166},
  isbn={979-8-4007-2258-5/2026/08}
}
```

## Acknowledgments

This work was funded in part by the European Union's Horizon Europe Research and Innovation Programme under the Marie Sklodowska-Curie under Grant 101131117 and UKRI under Grant reference number EP/Z000041/1.