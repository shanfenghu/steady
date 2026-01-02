import numpy as np

def mle_estimator(Y: np.ndarray) -> np.ndarray:
    """
    Computes the Maximum Likelihood Estimate (MLE) for a vector of means.

    Under the assumption of Gaussian noise, the MLE for the mean of each
    observation is simply the observation itself. This serves as the fundamental
    baseline for comparison with shrinkage estimators like GJS and STEADY.

    Args:
        Y (np.ndarray): A 1D numpy array of n noisy observations (Y_1, ..., Y_n).

    Returns:
        np.ndarray: The MLE estimate vector, which is identical to the input
                    observation vector Y. Returns a copy to prevent accidental
                    modification of the original array.
    
    Raises:
        ValueError: If the input Y is not a 1D numpy array.
    
    Example:
        >>> Y = np.array([1.0, 2.0, 3.0])
        >>> mle_estimator(Y)
        array([1., 2., 3.])
    """
    # --- Input Validation ---
    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
        raise ValueError("Input Y must be a 1D numpy array.")

    # For the MLE, the estimate is the observation itself.
    # We return a copy to prevent accidental modification of the original input array.
    return np.copy(Y)

def gjs_estimator(Y: np.ndarray, sigmas_sq: np.ndarray) -> np.ndarray:
    """
    Computes the Generalized James-Stein (GJS) estimator for means with
    unequal variances.

    This is a state-of-the-art, purely data-driven shrinkage estimator for the
    heteroscedastic case. It shrinks observations towards a precision-weighted
    mean of the data, providing a data-driven alternative to the MLE that
    typically achieves lower risk in high-dimensional settings.

    The estimator applies adaptive shrinkage based on the precision (inverse variance)
    of each observation, with the shrinkage factor determined by the data itself.

    Args:
        Y (np.ndarray): A 1D numpy array of n noisy observations (Y_1, ..., Y_n).
        sigmas_sq (np.ndarray): A 1D numpy array of n known, unequal observation
                                variances (sigma_1^2, ..., sigma_n^2). All values
                                must be positive.

    Returns:
        np.ndarray: The GJS estimate vector. For n < 4, returns the MLE (Y itself).
        
    Raises:
        ValueError: If inputs are not 1D numpy arrays, have mismatched shapes,
                    or if sigmas_sq contains non-positive values.
    
    Example:
        >>> Y = np.array([1.0, 2.0, 3.0, 4.0])
        >>> sigmas_sq = np.array([1.0, 1.0, 1.0, 1.0])
        >>> gjs_estimator(Y, sigmas_sq)
        array([...])  # Shrunk estimates
    """
    # --- Input Validation ---
    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
        raise ValueError("Input Y must be a 1D numpy array.")
    if not isinstance(sigmas_sq, np.ndarray) or sigmas_sq.ndim != 1:
        raise ValueError("Input sigmas_sq must be a 1D numpy array.")
    if Y.shape != sigmas_sq.shape:
        raise ValueError("Inputs Y and sigmas_sq must have the same shape.")
    if np.any(sigmas_sq <= 0):
        raise ValueError("All variances in sigmas_sq must be positive.")

    n = len(Y)
    if n < 4:
        # Shrinkage is only effective for n > 3 in this formulation.
        # For n <= 3, the GJS estimator defaults to the MLE.
        return np.copy(Y)

    # Calculate precision (inverse variance)
    precision = 1.0 / sigmas_sq

    # Step 1: Calculate the precision-weighted mean (the shrinkage target)
    y_bar_w = np.sum(Y * precision) / np.sum(precision)

    # Step 2: Calculate the precision-weighted sum of squared deviations
    S_j = np.sum(((Y - y_bar_w)**2) * precision)
    
    # Handle edge case where all observations are identical
    if S_j == 0:
        return np.copy(Y)

    # Step 3: Calculate the shrinkage constant
    # We use n-3 because one degree of freedom is used to estimate the mean.
    c_gjs = n - 3.0

    # Step 4: Calculate the shrinkage factor with the positive-part rule
    shrinkage_factor = 1.0 - (c_gjs / S_j)
    shrinkage_factor = max(0, shrinkage_factor)

    # Step 5: Apply shrinkage and compute the final estimate
    mu_hat_gjs = y_bar_w + shrinkage_factor * (Y - y_bar_w)

    return mu_hat_gjs

def steady_estimator(
    Y: np.ndarray,
    lambdas: np.ndarray,
    mu_phys: float,
    sigma_ou_sq: float,
    sigma_obs_sq: float
) -> np.ndarray:
    """
    Computes the STEADY (STein-type Estimator Assisted by DYnamics) estimate.

    This estimator generalizes Stein's paradox by shrinking observations towards
    a physical prior, with an adaptive shrinkage factor determined by the
    physical stability of each system. The amount of shrinkage for each observation
    is modulated by its physical reversion rate (lambda), with more stable systems
    (higher lambda) receiving less shrinkage.

    The estimator combines physical knowledge (via the prior mu_phys and stability
    parameters lambdas) with data-driven hyperparameter estimation to achieve
    improved performance over purely data-driven methods when physical information
    is available and reasonably accurate.

    Args:
        Y (np.ndarray): 1D array of n noisy observations (Y_1, ..., Y_n).
        lambdas (np.ndarray): 1D array of n physical reversion rates (lambda_i).
                             Higher values indicate more stable systems. All values
                             must be positive.
        mu_phys (float): The global physical mean, serving as the shrinkage target.
                         This represents the expected equilibrium value based on
                         physical knowledge.
        sigma_ou_sq (float): The variance of the physical process noise in the
                            underlying Ornstein-Uhlenbeck process. Must be non-negative.
        sigma_obs_sq (float): The variance of the measurement error. Must be non-negative.

    Returns:
        np.ndarray: The STEADY estimate vector. For n < 3, returns the MLE (Y itself).

    Raises:
        ValueError: If inputs have incorrect types, shapes, or non-positive values.
    
    Example:
        >>> Y = np.array([0.5, 5.2, 9.8])
        >>> lambdas = np.array([1.0, 0.5, 0.2])
        >>> mu_phys = 4.0
        >>> sigma_ou_sq = 2.0
        >>> sigma_obs_sq = 1.0
        >>> steady_estimator(Y, lambdas, mu_phys, sigma_ou_sq, sigma_obs_sq)
        array([...])  # Physics-informed shrunk estimates
    """
    # --- Input Validation ---
    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
        raise ValueError("Input Y must be a 1D numpy array.")
    if not isinstance(lambdas, np.ndarray) or lambdas.ndim != 1:
        raise ValueError("Input lambdas must be a 1D numpy array.")
    if Y.shape != lambdas.shape:
        raise ValueError("Inputs Y and lambdas must have the same shape.")
    if np.any(lambdas <= 0):
        raise ValueError("All reversion rates in lambdas must be positive.")
    if sigma_ou_sq < 0 or sigma_obs_sq < 0:
        raise ValueError("Variance components cannot be negative.")

    n = len(Y)
    if n < 3:
        # Dominance is only guaranteed for n >= 3. Default to MLE.
        return np.copy(Y)

    # Step 1: Calculate total heteroscedastic variance for each observation
    # The total variance combines physical process variance (which depends on
    # the stability parameter lambda) and measurement error variance.
    # Formula: sigma_i^2 = (sigma_ou^2 / (2 * lambda_i)) + sigma_obs^2
    sigmas_i_sq = (sigma_ou_sq / (2 * lambdas)) + sigma_obs_sq

    # Step 2: Estimate the hyperparameter tau^2 using the method of moments
    # This represents the variance of the true means around the physical prior.
    S = np.sum((Y - mu_phys)**2)
    tau_sq_hat = (S / n) - np.mean(sigmas_i_sq)
    
    # Apply the positive-part rule to ensure variance is non-negative
    # This prevents negative variance estimates that can occur due to sampling noise.
    tau_sq_hat = max(0, tau_sq_hat)

    # Step 3: Calculate the adaptive shrinkage factor for each observation
    # B_i represents how much to shrink towards the prior. Higher B_i means more shrinkage.
    # B_i = (Observation Variance) / (Observation Variance + Prior Variance)
    # Systems with higher stability (larger lambda) have lower observation variance,
    # leading to lower B_i and less shrinkage.
    B_i_hat = sigmas_i_sq / (sigmas_i_sq + tau_sq_hat)

    # Step 4: Compute the final STEADY estimate
    # The estimate is a weighted average between the observation and the physical prior.
    # Formula: mu_hat_i = (1 - B_i) * Y_i + B_i * mu_phys
    mu_hat_steady = (1 - B_i_hat) * Y + B_i_hat * mu_phys

    return mu_hat_steady