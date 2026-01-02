import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Import our custom library functions and utilities
from steady.estimators import gjs_estimator, steady_estimator
from plot_utils import setup_plot_style, STEADY_COLORS, save_figure

def run_experiment(n_dims_range: np.ndarray = None, n_repeats: int = 100):
    """
    Runs the computational scalability experiment.

    This experiment measures the wall-clock time to compute the GJS and STEADY
    estimates for an increasing number of dimensions (n). The goal is to verify
    that both estimators scale efficiently (approximately O(n)) with the problem size.

    Args:
        n_dims_range (np.ndarray, optional): The range of dimensions (n) to test.
                                             Defaults to a log-spaced range from
                                             100 to 100,000.
        n_repeats (int): The number of times to repeat the estimation for each
                         dimension to get a stable average time measurement.

    Returns:
        tuple: A tuple containing:
            - n_dims_range (np.ndarray): The dimensions tested.
            - avg_times (dict): A dictionary with keys 'gjs' and 'steady' containing
                                the average runtime (in seconds) for each estimator.
    """
    if n_dims_range is None:
        # Use a log-spaced range to see the scaling behavior clearly
        n_dims_range = np.logspace(2, 5, 15, dtype=int) # From 100 to 100,000

    # --- Data Structures to Store Results ---
    avg_times = {
        'gjs': np.zeros(len(n_dims_range)),
        'steady': np.zeros(len(n_dims_range))
    }

    # --- Main Simulation Loop ---
    print("Running Experiment: Computational Scalability...")
    for i, n in enumerate(tqdm(n_dims_range, desc="Simulating Dimensions")):
        # --- Pre-generate data for this dimension ---
        Y = np.random.randn(n)
        sigmas_sq = np.random.uniform(low=0.5, high=2.0, size=n)
        lambdas = np.random.uniform(low=0.1, high=2.0, size=n)
        mu_phys = 5.0
        sigma_ou_sq = 1.0
        sigma_obs_sq = 1.0

        # --- Time GJS Estimator ---
        gjs_times = []
        for _ in range(n_repeats):
            start_time = time.perf_counter()
            gjs_estimator(Y, sigmas_sq)
            end_time = time.perf_counter()
            gjs_times.append(end_time - start_time)
        avg_times['gjs'][i] = np.mean(gjs_times)

        # --- Time STEADY Estimator ---
        steady_times = []
        for _ in range(n_repeats):
            start_time = time.perf_counter()
            steady_estimator(Y, lambdas, mu_phys, sigma_ou_sq, sigma_obs_sq)
            end_time = time.perf_counter()
            steady_times.append(end_time - start_time)
        avg_times['steady'][i] = np.mean(steady_times)

    return n_dims_range, avg_times

def plot_results(n_dims_range, avg_times):
    """
    Generates and saves the log-log scalability plot for the experiment.
    
    Creates a log-log plot showing how the runtime of each estimator scales with
    the number of dimensions. A reference line for ideal O(n) scaling is included
    for comparison.
    
    Args:
        n_dims_range (np.ndarray): The dimensions tested.
        avg_times (dict): Dictionary with average runtime for each estimator.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the measured runtimes on a log-log scale
    ax.loglog(n_dims_range, avg_times['gjs'], color=STEADY_COLORS['gjs'], label='GJS', lw=2, marker='o', markersize=6)
    ax.loglog(n_dims_range, avg_times['steady'], color=STEADY_COLORS['steady'], label='STEADY', lw=2, marker='^', markersize=6)

    # Plot a reference line for perfect linear scaling, O(n)
    # We scale it to start near our first data point for easy comparison
    reference_slope = n_dims_range * (avg_times['steady'][0] / n_dims_range[0])
    ax.loglog(n_dims_range, reference_slope, color='black', label='Ideal O(n) Scaling', ls=':', lw=2.5)

    # --- Final Touches ---
    ax.set_xlabel("Number of Dimensions (n)")
    ax.set_ylabel("Average Runtime (seconds)")
    ax.set_title("Computational Scalability of Estimators")
    ax.legend(title="Estimator")
    ax.grid(True, which="both", ls="--")

    save_figure(fig, "exp_scalability")

if __name__ == "__main__":
    # Run the experiment and generate the plot
    dims, times = run_experiment()
    plot_results(dims, times)
