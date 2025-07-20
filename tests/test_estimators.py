import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

# Import the functions to be tested
from steady.estimators import mle_estimator, gjs_estimator, steady_estimator

# --- Fixtures for reusable test data ---

@pytest.fixture
def sample_data():
    """Provides a standard set of data for testing."""
    Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sigmas_sq = np.array([0.5, 0.8, 1.0, 1.2, 1.5])
    lambdas = np.array([1.0, 0.8, 0.5, 0.3, 0.1])
    return Y, sigmas_sq, lambdas

# --- Tests for mle_estimator ---

def test_mle_returns_correct_values(sample_data):
    """Tests that the MLE is identical to the observation vector."""
    Y, _, _ = sample_data
    mle_est = mle_estimator(Y)
    assert_array_equal(mle_est, Y)

def test_mle_returns_a_copy(sample_data):
    """Tests that the MLE returns a copy, not a view, of the input array."""
    Y, _, _ = sample_data
    mle_est = mle_estimator(Y)
    assert mle_est is not Y

def test_mle_invalid_input():
    """Tests that MLE raises ValueError for incorrect input types."""
    with pytest.raises(ValueError, match="Input Y must be a 1D numpy array."):
        mle_estimator([1, 2, 3]) # Should be a numpy array
    with pytest.raises(ValueError, match="Input Y must be a 1D numpy array."):
        mle_estimator(np.array([[1, 2], [3, 4]])) # Should be 1D

# --- Tests for gjs_estimator ---

def test_gjs_known_case():
    """Tests the GJS estimator against a pre-calculated, known result."""
    Y = np.array([1, 2, 8, 12])
    sigmas_sq = np.array([1, 1, 1, 1])
    expected_est = np.array([1.05882353, 2.04644118, 7.97213607, 11.92260022])
    gjs_est = gjs_estimator(Y, sigmas_sq)
    # Use a reasonable tolerance to account for floating point arithmetic.
    assert_allclose(gjs_est, expected_est, rtol=1e-6)

def test_gjs_no_shrinkage_for_n_less_than_4():
    """Tests that GJS defaults to MLE for n < 4."""
    Y = np.array([1, 2, 3])
    sigmas_sq = np.array([1, 1, 1])
    gjs_est = gjs_estimator(Y, sigmas_sq)
    assert_array_equal(gjs_est, Y)

def test_gjs_positive_part_rule():
    """Tests that the shrinkage factor is truncated at 0."""
    Y = np.array([3.9, 4.0, 4.1, 4.2])
    sigmas_sq = np.array([1, 1, 1, 1])
    y_bar_w = np.mean(Y)
    expected_est = np.full_like(Y, y_bar_w)
    gjs_est = gjs_estimator(Y, sigmas_sq)
    assert_allclose(gjs_est, expected_est)

def test_gjs_zero_deviation_edge_case():
    """Tests that GJS returns MLE if all observations are identical."""
    Y = np.array([5, 5, 5, 5])
    sigmas_sq = np.array([1, 1, 1, 1])
    gjs_est = gjs_estimator(Y, sigmas_sq)
    assert_array_equal(gjs_est, Y)

def test_gjs_invalid_inputs(sample_data):
    """Tests that GJS raises ValueErrors for various invalid inputs."""
    Y, sigmas_sq, _ = sample_data
    with pytest.raises(ValueError, match="must have the same shape"):
        gjs_estimator(Y, sigmas_sq[:4])
    with pytest.raises(ValueError, match="must be positive"):
        gjs_estimator(Y, np.array([1, 1, 0, 1, 1]))
    with pytest.raises(ValueError, match="must be positive"):
        gjs_estimator(Y, np.array([1, 1, -1, 1, 1]))

# --- Tests for steady_estimator ---

def test_steady_known_case():
    """Tests the STEADY estimator against a pre-calculated, known result."""
    Y = np.array([0, 5, 10])
    lambdas = np.array([1.0, 0.5, 0.2])
    mu_phys, sigma_ou_sq, sigma_obs_sq = 4.0, 2.0, 1.0
    expected_est = np.array([0.5, 4.82352941, 8.2])
    steady_est = steady_estimator(Y, lambdas, mu_phys, sigma_ou_sq, sigma_obs_sq)
    assert_allclose(steady_est, expected_est, rtol=1e-3)

def test_steady_no_shrinkage_for_n_less_than_3():
    """Tests that STEADY defaults to MLE for n < 3."""
    Y = np.array([1, 10])
    lambdas = np.array([1, 0.1])
    steady_est = steady_estimator(Y, lambdas, 5.0, 1.0, 1.0)
    assert_array_equal(steady_est, Y)

def test_steady_positive_part_tau_rule():
    """Tests that tau_sq_hat is truncated at 0."""
    Y = np.array([3.9, 4.0, 4.1])
    lambdas = np.array([1, 1, 1])
    mu_phys, sigma_ou_sq, sigma_obs_sq = 4.0, 20.0, 1.0
    expected_est = np.full_like(Y, mu_phys)
    steady_est = steady_estimator(Y, lambdas, mu_phys, sigma_ou_sq, sigma_obs_sq)
    assert_allclose(steady_est, expected_est)

def test_steady_adaptive_shrinkage_logic():
    """Tests that higher stability (lambda) leads to less shrinkage."""
    Y = np.array([0, 5, 10])
    mu_phys = 5.0
    lambdas_high = np.array([10.0, 10.0, 10.0])
    est_high_lambda = steady_estimator(Y, lambdas_high, mu_phys, 1.0, 1.0)
    lambdas_low = np.array([0.1, 0.1, 0.1])
    est_low_lambda = steady_estimator(Y, lambdas_low, mu_phys, 1.0, 1.0)
    
    dist_high_lambda = np.linalg.norm(est_high_lambda - Y)
    dist_low_lambda = np.linalg.norm(est_low_lambda - Y)
    assert dist_high_lambda < dist_low_lambda

def test_steady_invalid_inputs(sample_data):
    """Tests that STEADY raises ValueErrors for various invalid inputs."""
    Y, _, lambdas = sample_data
    with pytest.raises(ValueError, match="must have the same shape"):
        steady_estimator(Y, lambdas[:4], 5.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="must be positive"):
        steady_estimator(Y, np.array([1, 1, 0, 1, 1]), 5.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        steady_estimator(Y, lambdas, 5.0, -1.0, 1.0)
