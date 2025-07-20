import numpy as np
from typing import Tuple, Union

def simulate_ou_process(
    n: int,
    mu_true: Union[float, np.ndarray],
    lambdas: Union[float, np.ndarray],
    sigma_ou: float,
    sigma_obs: float,
    t_end: float = 100.0,
    dt: float = 0.1,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates n independent Ornstein-Uhlenbeck processes to generate a snapshot
    of noisy observations from their stationary distribution.

    This function serves as the data generator for all synthetic experiments.

    Args:
        n (int): The number of systems/sensors to simulate (n >= 3).
        mu_true (Union[float, np.ndarray]): The true equilibrium parameter(s).
            If a float, all systems share the same equilibrium.
            If an array, it must have length n.
        lambdas (Union[float, np.ndarray]): The physical reversion rate(s).
            If a float, all systems share the same stability.
            If an array, it must have length n.
        sigma_ou (float): The standard deviation of the physical process noise.
        sigma_obs (float): The standard deviation of the measurement error.
        t_end (float, optional): The total simulation time to ensure the process
                                 reaches its stationary state. Defaults to 100.0.
        dt (float, optional): The time step for the simulation. Defaults to 0.1.
        seed (int, optional): A random seed for reproducibility. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - Y (np.ndarray): The final vector of n noisy observations.
            - mu_true_vec (np.ndarray): The vector of n true equilibrium values.
            - sigmas_i_sq (np.ndarray): The vector of n total observation variances.
            
    Raises:
        ValueError: If inputs have incorrect types, shapes, or non-positive values.
    """
    # --- Input Validation ---
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")
    if sigma_ou < 0 or sigma_obs < 0 or dt <= 0 or t_end <= 0:
        raise ValueError("Variances, dt, and t_end must be positive.")

    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # --- Prepare parameter vectors ---
    # Ensure mu_true and lambdas are numpy arrays of the correct size
    if isinstance(mu_true, (int, float)):
        mu_true_vec = np.full(n, float(mu_true))
    elif isinstance(mu_true, np.ndarray) and mu_true.shape == (n,):
        mu_true_vec = mu_true
    else:
        raise ValueError("mu_true must be a float or a numpy array of shape (n,).")

    if isinstance(lambdas, (int, float)):
        lambdas_vec = np.full(n, float(lambdas))
    elif isinstance(lambdas, np.ndarray) and lambdas.shape == (n,):
        lambdas_vec = lambdas
    else:
        raise ValueError("lambdas must be a float or a numpy array of shape (n,).")
        
    if np.any(lambdas_vec <= 0):
        raise ValueError("All reversion rates in lambdas must be positive.")

    # --- Simulate the OU Process using Euler-Maruyama method ---
    num_steps = int(t_end / dt)
    X = np.zeros((n, num_steps))  # n paths, num_steps long

    for t in range(1, num_steps):
        # Generate random noise for all n paths at once
        dW = np.random.normal(0, np.sqrt(dt), n)
        # Update all n paths
        X[:, t] = X[:, t-1] + lambdas_vec * (mu_true_vec - X[:, t-1]) * dt + sigma_ou * dW

    # The final state of the physical process is the last time step
    X_final = X[:, -1]

    # --- Generate the Final Observation ---
    # Add measurement error to the final physical state
    measurement_noise = np.random.normal(0, sigma_obs, n)
    Y = X_final + measurement_noise

    # --- Calculate the total observation variance for each system ---
    sigma_ou_sq = sigma_ou**2
    sigma_obs_sq = sigma_obs**2
    sigmas_i_sq = (sigma_ou_sq / (2 * lambdas_vec)) + sigma_obs_sq

    return Y, mu_true_vec, sigmas_i_sq
