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
    Simulates n independent Ornstein-Uhlenbeck (OU) processes to generate a snapshot
    of noisy observations from their stationary distribution.

    The Ornstein-Uhlenbeck process is a mean-reverting stochastic process commonly
    used to model physical systems that tend towards an equilibrium. Each process
    evolves according to:
        dX_t = lambda * (mu_true - X_t) * dt + sigma_ou * dW_t
    
    where lambda is the reversion rate, mu_true is the equilibrium, and dW_t is
    Brownian motion. After sufficient time, the process reaches a stationary
    distribution with mean mu_true and variance sigma_ou^2 / (2 * lambda).

    This function serves as the data generator for all synthetic experiments,
    simulating the physical process and then adding measurement noise.

    Args:
        n (int): The number of systems/sensors to simulate. Must be positive.
        mu_true (Union[float, np.ndarray]): The true equilibrium parameter(s).
            If a float, all systems share the same equilibrium.
            If an array, it must have length n.
        lambdas (Union[float, np.ndarray]): The physical reversion rate(s).
            Higher values indicate faster reversion to equilibrium (more stable).
            If a float, all systems share the same stability.
            If an array, it must have length n. All values must be positive.
        sigma_ou (float): The standard deviation of the physical process noise.
                          Must be non-negative.
        sigma_obs (float): The standard deviation of the measurement error.
                           Must be non-negative.
        t_end (float, optional): The total simulation time to ensure the process
                                 reaches its stationary state. Defaults to 100.0.
                                 Must be positive.
        dt (float, optional): The time step for the Euler-Maruyama discretization.
                              Defaults to 0.1. Must be positive.
        seed (int, optional): A random seed for reproducibility. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - Y (np.ndarray): The final vector of n noisy observations (physical
                              state + measurement error).
            - mu_true_vec (np.ndarray): The vector of n true equilibrium values.
            - sigmas_i_sq (np.ndarray): The vector of n total observation variances,
                                        computed as sigma_ou^2 / (2 * lambda_i) + sigma_obs^2.
            
    Raises:
        ValueError: If inputs have incorrect types, shapes, or non-positive values.
    
    Example:
        >>> Y, mu_true, sigmas_sq = simulate_ou_process(
        ...     n=10, mu_true=5.0, lambdas=1.0, 
        ...     sigma_ou=1.5, sigma_obs=1.0, seed=42
        ... )
        >>> print(Y.shape)
        (10,)
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
    # The Euler-Maruyama method is a numerical scheme for solving stochastic
    # differential equations. We discretize the continuous-time OU process
    # into discrete time steps.
    num_steps = int(t_end / dt)
    X = np.zeros((n, num_steps))  # n independent paths, each with num_steps time points

    for t in range(1, num_steps):
        # Generate random Brownian increments for all n paths simultaneously
        # The variance of dW is dt (standard deviation is sqrt(dt))
        dW = np.random.normal(0, np.sqrt(dt), n)
        # Update all n paths according to the OU process dynamics:
        # X_t = X_{t-1} + lambda * (mu_true - X_{t-1}) * dt + sigma_ou * dW
        X[:, t] = X[:, t-1] + lambdas_vec * (mu_true_vec - X[:, t-1]) * dt + sigma_ou * dW

    # The final state of the physical process is the last time step
    # After sufficient simulation time (t_end), the process has reached its
    # stationary distribution.
    X_final = X[:, -1]

    # --- Generate the Final Observation ---
    # Add measurement error to the final physical state to simulate realistic
    # sensor observations. The measurement error is independent Gaussian noise.
    measurement_noise = np.random.normal(0, sigma_obs, n)
    Y = X_final + measurement_noise

    # --- Calculate the total observation variance for each system ---
    # The total variance combines the stationary variance of the OU process
    # (which depends on lambda) and the measurement error variance.
    # Formula: sigma_i^2 = (sigma_ou^2 / (2 * lambda_i)) + sigma_obs^2
    sigma_ou_sq = sigma_ou**2
    sigma_obs_sq = sigma_obs**2
    sigmas_i_sq = (sigma_ou_sq / (2 * lambdas_vec)) + sigma_obs_sq

    return Y, mu_true_vec, sigmas_i_sq
