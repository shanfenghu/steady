import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our library functions and utilities
from steady.simulation import simulate_ou_process
from steady.estimators import mle_estimator, gjs_estimator, steady_estimator
from plot_utils import setup_plot_style, STEADY_COLORS, save_figure

def run_experiment(n_trials: int = 100, n_dims_range: list = None):
    """
    Runs the simulation experiment to verify dominance over MLE.

    This experiment calculates the risk (mean squared error) for the MLE, GJS, and STEADY
    estimators across a range of dimensions (n). The risk is computed as the average
    squared error between the estimates and the true equilibrium values.

    Args:
        n_trials (int): The number of independent simulations to run for each
                        dimension to get a stable average risk estimate.
        n_dims_range (list, optional): The range of dimensions (n) to test.
                                       Defaults to range(3, 51).

    Returns:
        tuple: A tuple containing:
            - n_dims_range (list): The dimensions tested.
            - mean_risks (dict): A dictionary with keys 'mle', 'gjs', 'steady' containing
                                 the mean risk (MSE) for each estimator.
            - std_errors (dict): A dictionary with keys 'mle', 'gjs', 'steady' containing
                                 the standard error of the mean risk for each estimator.
    """
    if n_dims_range is None:
        n_dims_range = list(range(3, 51))

    # --- Simulation Parameters ---
    mu_true_mean = 5.0
    mu_phys = 5.5  # A slightly misspecified physical prior
    sigma_ou = 1.5
    sigma_obs = 1.0
    lambda_min, lambda_max = 0.2, 2.0

    # --- Data Structures to Store Results ---
    results = {
        'mle': np.zeros((len(n_dims_range), n_trials)),
        'gjs': np.zeros((len(n_dims_range), n_trials)),
        'steady': np.zeros((len(n_dims_range), n_trials))
    }

    # --- Main Simulation Loop ---
    print("Running Experiment: Dominance over MLE...")
    for i, n in enumerate(tqdm(n_dims_range, desc="Simulating Dimensions")):
        for trial in range(n_trials):
            # For each trial, generate a new random problem
            mu_true = np.random.normal(loc=mu_true_mean, scale=2.0, size=n)
            lambdas = np.random.uniform(low=lambda_min, high=lambda_max, size=n)

            # Simulate the observed data
            Y, mu_true_vec, sigmas_i_sq = simulate_ou_process(
                n=n, mu_true=mu_true, lambdas=lambdas,
                sigma_ou=sigma_ou, sigma_obs=sigma_obs
            )

            # Get estimates from all three methods
            est_mle = mle_estimator(Y)
            est_gjs = gjs_estimator(Y, sigmas_i_sq)
            est_steady = steady_estimator(
                Y, lambdas, mu_phys, sigma_ou**2, sigma_obs**2
            )

            # Calculate and store the risk (MSE) for each
            results['mle'][i, trial] = np.mean((est_mle - mu_true_vec)**2)
            results['gjs'][i, trial] = np.mean((est_gjs - mu_true_vec)**2)
            results['steady'][i, trial] = np.mean((est_steady - mu_true_vec)**2)

    # --- Calculate Final Statistics ---
    mean_risks = {key: np.mean(val, axis=1) for key, val in results.items()}
    std_errors = {key: np.std(val, axis=1) / np.sqrt(n_trials) for key, val in results.items()}

    return n_dims_range, mean_risks, std_errors

def plot_results(n_dims_range, mean_risks, std_errors):
    """
    Generates and saves the risk vs. dimension plot for the experiment.
    
    Creates a line plot showing how the risk (MSE) of each estimator varies
    with the number of dimensions, with shaded error bars representing
    standard errors.
    
    Args:
        n_dims_range (list): The dimensions tested.
        mean_risks (dict): Dictionary with mean risk for each estimator.
        std_errors (dict): Dictionary with standard error for each estimator.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the mean risk for each estimator
    ax.plot(n_dims_range, mean_risks['mle'], color=STEADY_COLORS['mle'], label='MLE', lw=2)
    ax.plot(n_dims_range, mean_risks['gjs'], color=STEADY_COLORS['gjs'], label='GJS', lw=2)
    ax.plot(n_dims_range, mean_risks['steady'], color=STEADY_COLORS['steady'], label='STEADY', lw=2)

    # Add shaded error bars (mean +/- 1 standard error)
    ax.fill_between(n_dims_range,
                    mean_risks['mle'] - std_errors['mle'],
                    mean_risks['mle'] + std_errors['mle'],
                    color=STEADY_COLORS['mle'], alpha=0.2)
    ax.fill_between(n_dims_range,
                    mean_risks['gjs'] - std_errors['gjs'],
                    mean_risks['gjs'] + std_errors['gjs'],
                    color=STEADY_COLORS['gjs'], alpha=0.2)
    ax.fill_between(n_dims_range,
                    mean_risks['steady'] - std_errors['steady'],
                    mean_risks['steady'] + std_errors['steady'],
                    color=STEADY_COLORS['steady'], alpha=0.2)

    # --- Final Touches ---
    ax.set_xlabel("Number of Dimensions (n)")
    ax.set_ylabel("Risk (Mean Squared Error)")
    ax.set_title("Risk vs. Dimension: Verifying Dominance over MLE")
    ax.legend(title="Estimator")
    ax.grid(True, which="both", ls="--")
    
    # Set a y-limit to better visualize the difference
    min_risk = min(np.min(mean_risks['gjs']), np.min(mean_risks['steady']))
    max_risk = np.max(mean_risks['mle'])
    ax.set_ylim(min_risk * 0.9, max_risk * 1.1)

    save_figure(fig, "exp_dominance_vs_mle")

if __name__ == "__main__":
    # Run the experiment and generate the plot
    n_dims, means, errors = run_experiment()
    plot_results(n_dims, means, errors)
