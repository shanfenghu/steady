import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

# Import our library functions and utilities
from steady.simulation import simulate_ou_process
from steady.estimators import gjs_estimator, steady_estimator
from plot_utils import setup_plot_style, STEADY_COLORS, save_figure

def run_experiment_3(
    n_trials: int = 100,
    n_dims: int = 20,
    lambda_bias_factors: np.ndarray = None,
    mu_phys_errors: list = None
):
    """
    Runs a revised, more rigorous simulation for the experiment, testing
    sensitivity to a systematic bias in the lambda parameter.

    Args:
        n_trials (int): Number of independent simulations for each setting.
        n_dims (int): Fixed number of dimensions for the simulation.
        lambda_bias_factors (np.ndarray, optional): Multiplicative bias factors for lambda.
        mu_phys_errors (list, optional): Different error levels for the physical prior.

    Returns:
        tuple: A tuple containing the results for plotting.
    """
    if lambda_bias_factors is None:
        # Test underestimation (0.1) to overestimation (10.0)
        lambda_bias_factors = np.logspace(-1, 1, 21)
    if mu_phys_errors is None:
        mu_phys_errors = [0.2, 0.4, 0.8, 1.2, 1.6]

    # --- Simulation Parameters ---
    mu_true_mean = 5.0
    sigma_ou = 3.0
    sigma_obs = 2.0

    # --- Data Structures to Store Results ---
    results = {
        'gjs': np.zeros((len(lambda_bias_factors), n_trials)),
        'steady': {err: np.zeros((len(lambda_bias_factors), n_trials)) for err in mu_phys_errors}
    }

    # --- Main Simulation Loop ---
    print("Running Experiment: Sensitivity Analysis...")
    param_combinations = list(itertools.product(enumerate(lambda_bias_factors), enumerate(mu_phys_errors)))

    for (i, bias_factor), (j, mu_error) in tqdm(param_combinations, desc="Simulating Bias Factors"):
        mu_phys = mu_true_mean + mu_error

        for trial in range(n_trials):
            true_lambdas = np.random.uniform(low=0.5, high=2.5, size=n_dims)
            mu_true = np.random.normal(loc=mu_true_mean, scale=0.5, size=n_dims)

            # Introduce a systematic bias to the lambdas
            biased_lambdas = true_lambdas * bias_factor

            # Simulate data using the TRUE lambdas
            Y, mu_true_vec, sigmas_i_sq_true = simulate_ou_process(
                n=n_dims, mu_true=mu_true, lambdas=true_lambdas,
                sigma_ou=sigma_ou, sigma_obs=sigma_obs
            )

            # Get estimates
            est_gjs = gjs_estimator(Y, sigmas_i_sq_true)
            # Provide the BIASED lambdas to the STEADY estimator
            est_steady = steady_estimator(
                Y, biased_lambdas, mu_phys, sigma_ou**2, sigma_obs**2
            )

            # Calculate and store the risk
            results['gjs'][i, trial] = np.mean((est_gjs - mu_true_vec)**2)
            results['steady'][mu_error][i, trial] = np.mean((est_steady - mu_true_vec)**2)

    # --- Calculate Final Statistics ---
    mean_risks_gjs = np.mean(results['gjs'], axis=1)
    std_errors_gjs = np.std(results['gjs'], axis=1) / np.sqrt(n_trials)

    mean_risks_steady = {err: np.mean(res, axis=1) for err, res in results['steady'].items()}
    std_errors_steady = {err: np.std(res, axis=1) / np.sqrt(n_trials) for err, res in results['steady'].items()}

    return lambda_bias_factors, mu_phys_errors, mean_risks_gjs, std_errors_gjs, mean_risks_steady, std_errors_steady

def plot_results(lambda_bias_factors, mu_phys_errors, mean_risks_gjs, std_errors_gjs, mean_risks_steady, std_errors_steady):
    """
    Generates and saves the revised plot for the experiment.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot GJS as a single horizontal baseline
    gjs_baseline_risk = np.mean(mean_risks_gjs)
    ax.axhline(y=gjs_baseline_risk, color=STEADY_COLORS['gjs'], label='GJS (Data-Driven Baseline)', ls='--', lw=2.5)

    # Use a color gradient for the different STEADY curves
    color_map = plt.cm.get_cmap('viridis_r', len(mu_phys_errors) + 2)

    for i, mu_error in enumerate(mu_phys_errors):
        label = f'STEADY (Prior Error = {mu_error})'
        ax.plot(lambda_bias_factors, mean_risks_steady[mu_error], color=color_map(i / len(mu_phys_errors)),
                label=label, lw=2, marker='o', markersize=5, alpha=0.8)
        ax.fill_between(lambda_bias_factors,
                        mean_risks_steady[mu_error] - std_errors_steady[mu_error],
                        mean_risks_steady[mu_error] + std_errors_steady[mu_error],
                        color=color_map(i / len(mu_phys_errors)), alpha=0.15)

    # --- Final Touches and Annotations ---
    ax.set_xscale('log') # Use a log scale for the bias factor
    ax.axvline(x=1.0, color='black', ls=':', lw=2, label='Correct $\lambda$ (Bias Factor = 1.0)')
    ax.set_xlabel("Bias Factor for Stability $\lambda$ (Multiplicative, log scale)")
    ax.set_ylabel("Risk (Mean Squared Error)")
    ax.set_title("STEADY's Robustness to Interacting Parameter Errors")
    
    handles, labels = ax.get_legend_handles_labels()
    order = [0, len(mu_phys_errors) + 1] + list(range(1, len(mu_phys_errors) + 1))
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], title="Estimator")

    ax.grid(True, which="both", ls="--")
    save_figure(fig, "exp_sensitivity")

if __name__ == "__main__":
    l_bias, m_errs, gjs_risk, gjs_std, steady_risks, steady_stds = run_experiment_3()
    plot_results(l_bias, m_errs, gjs_risk, gjs_std, steady_risks, steady_stds)
