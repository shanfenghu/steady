import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our library functions and utilities
from steady.simulation import simulate_ou_process
from steady.estimators import gjs_estimator, steady_estimator
from plot_utils import setup_plot_style, STEADY_COLORS, save_figure

def run_experiment_2(n_trials: int = 100, n_dims: int = 20, error_levels: np.ndarray = None):
    """
    Runs the simulation experiment to verify the model correctness condition.

    This experiment calculates the risk (mean squared error) for GJS and STEADY
    estimators as the physical prior becomes progressively less accurate. The
    error level represents the offset between the true mean and the physical prior,
    allowing us to study how robust STEADY is to prior misspecification.

    Args:
        n_trials (int): The number of independent simulations for each error level.
        n_dims (int): The fixed number of dimensions for the simulation.
        error_levels (np.ndarray, optional): The range of misspecification errors
                                             to test. Defaults to np.linspace(0, 1.5, 25).

    Returns:
        tuple: A tuple containing:
            - error_levels (np.ndarray): The misspecification levels tested.
            - mean_risks (dict): A dictionary with keys 'gjs' and 'steady' containing
                                 the mean risk (MSE) for each estimator.
            - std_errors (dict): A dictionary with keys 'gjs' and 'steady' containing
                                 the standard error of the mean risk for each estimator.
    """
    if error_levels is None:
        error_levels = np.linspace(0, 1.5, 25)

    # --- Simulation Parameters ---
    mu_true_mean = 5.0
    sigma_ou = 3.0
    sigma_obs = 2.0
    lambdas = 1.0  # Use uniform stability for this experiment

    # --- Data Structures to Store Results ---
    results = {
        'gjs': np.zeros((len(error_levels), n_trials)),
        'steady': np.zeros((len(error_levels), n_trials))
    }

    # --- Main Simulation Loop ---
    print("Running Experiment: Model Correctness Condition...")
    for i, error in enumerate(tqdm(error_levels, desc="Simulating Error Levels")):
        # The physical prior is offset from the true mean by the error level
        mu_phys = mu_true_mean + error

        for trial in range(n_trials):
            # Generate a new true mean vector for each trial
            mu_true = np.random.normal(loc=mu_true_mean, scale=0.1, size=n_dims)

            # Simulate the observed data
            Y, mu_true_vec, sigmas_i_sq = simulate_ou_process(
                n=n_dims, mu_true=mu_true, lambdas=lambdas,
                sigma_ou=sigma_ou, sigma_obs=sigma_obs
            )

            # Get estimates from GJS and STEADY
            est_gjs = gjs_estimator(Y, sigmas_i_sq)
            est_steady = steady_estimator(
                Y, np.full(n_dims, lambdas), mu_phys, sigma_ou**2, sigma_obs**2
            )

            # Calculate and store the risk (MSE) for each
            results['gjs'][i, trial] = np.mean((est_gjs - mu_true_vec)**2)
            results['steady'][i, trial] = np.mean((est_steady - mu_true_vec)**2)

    # --- Calculate Final Statistics ---
    mean_risks = {key: np.mean(val, axis=1) for key, val in results.items()}
    std_errors = {key: np.std(val, axis=1) / np.sqrt(n_trials) for key, val in results.items()}

    return error_levels, mean_risks, std_errors

def plot_results(error_levels, mean_risks, std_errors):
    """
    Generates and saves the risk vs. prior accuracy plot for the experiment.
    
    Creates a line plot showing how the risk (MSE) of each estimator varies
    with the misspecification error of the physical prior, with shaded error bars.
    Also annotates the crossover point where GJS becomes better than STEADY.
    
    Args:
        error_levels (np.ndarray): The misspecification levels tested.
        mean_risks (dict): Dictionary with mean risk for each estimator.
        std_errors (dict): Dictionary with standard error for each estimator.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the mean risk for each estimator
    ax.plot(error_levels, mean_risks['gjs'], color=STEADY_COLORS['gjs'], label='GJS (Data-Driven)', lw=2)
    ax.plot(error_levels, mean_risks['steady'], color=STEADY_COLORS['steady'], label='STEADY (Physics-Informed)', lw=2)

    # Add shaded error bars
    ax.fill_between(error_levels,
                    mean_risks['gjs'] - std_errors['gjs'],
                    mean_risks['gjs'] + std_errors['gjs'],
                    color=STEADY_COLORS['gjs'], alpha=0.2)
    ax.fill_between(error_levels,
                    mean_risks['steady'] - std_errors['steady'],
                    mean_risks['steady'] + std_errors['steady'],
                    color=STEADY_COLORS['steady'], alpha=0.2)

    # --- Final Touches and Annotations ---
    ax.set_xlabel("Model Misspecification Error ($||\mu_{phys} - \mu_{true}||$)")
    ax.set_ylabel("Risk (Mean Squared Error)")
    ax.set_title("Risk vs. Physical Prior Accuracy (Model Correctness)")
    ax.legend(title="Estimator")
    ax.grid(True, which="both", ls="--")

    # Find and annotate the crossover point
    crossover_idx = np.where(mean_risks['steady'] > mean_risks['gjs'])[0]
    if len(crossover_idx) > 0:
        crossover_x = error_levels[crossover_idx[0]]
        ax.axvline(crossover_x, color='black', ls=':', lw=2, label='Crossover Point')
        ax.text(crossover_x - 0.1, ax.get_ylim()[1] * 0.9, 'STEADY is better',
                ha='right', fontsize=12, color=STEADY_COLORS['steady'])
        ax.text(crossover_x + 0.1, ax.get_ylim()[1] * 0.9, 'GJS is better',
                ha='left', fontsize=12, color=STEADY_COLORS['gjs'])
        # Re-order legend to show axvline
        handles, labels = ax.get_legend_handles_labels()
        order = [0, 1, 2]
        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], title="Estimator")


    save_figure(fig, "exp_model_correctness")

if __name__ == "__main__":
    # Run the experiment and generate the plot
    errors, means, std_errs = run_experiment_2()
    plot_results(errors, means, std_errs)
