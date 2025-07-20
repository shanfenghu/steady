import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

# Import the function to be tested
from steady.simulation import simulate_ou_process

# --- Fixtures for reusable test parameters ---

@pytest.fixture
def sim_params():
    """Provides a standard set of simulation parameters."""
    return {
        "n": 5,
        "mu_true": 2.0,
        "lambdas": 0.5,
        "sigma_ou": 1.0,
        "sigma_obs": 0.5,
        "seed": 42
    }

# --- Tests for simulate_ou_process ---

def test_output_properties(sim_params):
    """Tests that the outputs have the correct shapes and types."""
    Y, mu_true_vec, sigmas_i_sq = simulate_ou_process(**sim_params)

    # Check types
    assert isinstance(Y, np.ndarray)
    assert isinstance(mu_true_vec, np.ndarray)
    assert isinstance(sigmas_i_sq, np.ndarray)

    # Check shapes
    assert Y.shape == (sim_params["n"],)
    assert mu_true_vec.shape == (sim_params["n"],)
    assert sigmas_i_sq.shape == (sim_params["n"],)

def test_reproducibility_with_seed(sim_params):
    """Tests that using the same seed produces identical results."""
    # Run the simulation twice with the same seed
    Y1, _, _ = simulate_ou_process(**sim_params)
    Y2, _, _ = simulate_ou_process(**sim_params)
    assert_array_equal(Y1, Y2)

    # Run with a different seed and ensure the results are different
    sim_params_different_seed = sim_params.copy()
    sim_params_different_seed["seed"] = 123
    Y3, _, _ = simulate_ou_process(**sim_params_different_seed)
    assert not np.array_equal(Y1, Y3)

def test_long_term_mean():
    """
    Tests that the mean of many simulations is close to the true mean.
    This is a statistical test.
    """
    n_sims = 1000  # Number of snapshots to average
    n_sensors = 10
    mu_true = 5.0
    
    # Generate many independent snapshots
    all_Y = np.array([
        simulate_ou_process(n=n_sensors, mu_true=mu_true, lambdas=0.2, sigma_ou=1.0, sigma_obs=0.5)[0]
        for _ in range(n_sims)
    ])
    
    # The mean of all observations should be very close to mu_true
    observed_mean = np.mean(all_Y)
    assert_allclose(observed_mean, mu_true, atol=0.1) # atol for statistical tolerance

def test_long_term_variance():
    """
    Tests that the variance of many simulations is close to the theoretical variance.
    This is a statistical test.
    """
    n_sims = 1000
    n_sensors = 10
    mu_true = 0.0 # Mean of 0 simplifies variance calculation
    lambdas = 0.5
    sigma_ou = 1.0
    sigma_obs = 0.5

    # Generate many independent snapshots
    all_Y = np.array([
        simulate_ou_process(n=n_sensors, mu_true=mu_true, lambdas=lambdas, sigma_ou=sigma_ou, sigma_obs=sigma_obs)[0]
        for _ in range(n_sims)
    ])

    # Theoretical total variance
    theoretical_variance = (sigma_ou**2 / (2 * lambdas)) + sigma_obs**2
    
    # Observed variance
    observed_variance = np.var(all_Y)
    assert_allclose(observed_variance, theoretical_variance, rtol=0.1) # rtol for statistical tolerance

def test_invalid_inputs():
    """Tests that the function raises ValueErrors for various invalid inputs."""
    # Test bad n
    with pytest.raises(ValueError, match="n must be a positive integer"):
        simulate_ou_process(n=0, mu_true=1, lambdas=1, sigma_ou=1, sigma_obs=1)

    # Test bad lambdas
    with pytest.raises(ValueError, match="All reversion rates in lambdas must be positive"):
        simulate_ou_process(n=3, mu_true=1, lambdas=np.array([1, 0, 1]), sigma_ou=1, sigma_obs=1)

    # Test bad variances
    with pytest.raises(ValueError, match="Variances, dt, and t_end must be positive"):
        simulate_ou_process(n=3, mu_true=1, lambdas=1, sigma_ou=-1, sigma_obs=1)

    # Test mismatched shapes
    with pytest.raises(ValueError, match="mu_true must be a float or a numpy array of shape"):
        simulate_ou_process(n=3, mu_true=np.array([1, 1]), lambdas=1, sigma_ou=1, sigma_obs=1)
