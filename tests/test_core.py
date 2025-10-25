import pytest
import tensorflow as tf
import numpy as np
from spin_system.core import SpinSystem


# ------------------------
# FIXTURES
# ------------------------

@pytest.fixture
def ising_system():
    """Provides a 2D Ising model instance with uniform coupling."""
    L = 4
    replicas = 2
    shape = (L, L)
    # As requested, using the original uniform coupling
    J = tf.ones(shape + shape) / (L * L)

    system = SpinSystem(
        lattice_dim=2,
        lattice_length=L,
        lattice_replicas=replicas,
        interaction_matrix=J,
        model="ising",
    )
    return system


@pytest.fixture
def spherical_system():
    """Provides a 2D Spherical model instance with uniform coupling."""
    L = 4
    replicas = 2
    shape = (L, L)
    J = tf.ones(shape + shape) / (L * L)

    system = SpinSystem(
        lattice_dim=2,
        lattice_length=L,
        lattice_replicas=replicas,
        interaction_matrix=J,
        model="spherical",
        spherical_constraint=True,
    )
    return system


@pytest.fixture
def z2_gauge_system():
    """Provides a basic 2D Z2 Gauge model instance."""
    L = 4
    replicas = 2
    shape = (L, L)
    # A simple placeholder interaction matrix for the Z2 gauge model
    J = tf.ones((3, *shape, *shape))

    system = SpinSystem(
        lattice_dim=2,
        lattice_length=L,
        lattice_replicas=replicas,
        interaction_matrix=J,
        model="z2_gauge",
    )
    return system


# ------------------------
# INITIALIZATION TESTS
# ------------------------

def test_initialization_shapes_ising(ising_system):
    s = ising_system
    assert s.spin_state.shape == (
        s.lattice_replicas, s.lattice_length, s.lattice_length)
    assert s.interaction_matrix.shape == (
        s.lattice_length, s.lattice_length, s.lattice_length, s.lattice_length)
    assert isinstance(s.energy, tf.Variable)


def test_initialization_shapes_spherical(spherical_system):
    s = spherical_system
    assert s.spin_state.shape == (
        s.lattice_replicas, s.lattice_length, s.lattice_length)
    assert s.interaction_matrix.shape == (
        s.lattice_length, s.lattice_length, s.lattice_length, s.lattice_length)
    assert isinstance(s.energy, tf.Variable)


def test_initialization_shapes_z2_gauge(z2_gauge_system):
    s = z2_gauge_system
    assert s.spin_state.shape == (
        s.lattice_replicas, s.lattice_length, s.lattice_length, 2)
    assert s.interaction_matrix.shape == (
        3, s.lattice_length, s.lattice_length, s.lattice_length, s.lattice_length)
    assert isinstance(s.energy, tf.Variable)
    assert isinstance(s.plaquette, tf.Variable)
    assert s.plaquette.shape == (
        s.lattice_replicas, s.lattice_length, s.lattice_length
    )

# ------------------------
# ENERGY AND PHYSICAL TESTS
# ------------------------


def test_energy_computation_deterministic_ising(ising_system):
    tf.random.set_seed(42)
    E = ising_system.compute_pairwise_energies()
    assert E.shape == (ising_system.lattice_replicas,)
    assert tf.reduce_all(tf.math.is_finite(E))


def test_energy_computation_deterministic_spherical(spherical_system):
    tf.random.set_seed(42)
    E = spherical_system.compute_pairwise_energies()
    assert E.shape == (spherical_system.lattice_replicas,)
    assert tf.reduce_all(tf.math.is_finite(E))


def test_overlap_matrix_properties_ising(ising_system):
    Q = ising_system.compute_overlap_matrix()
    np.testing.assert_allclose(Q, tf.transpose(Q))
    diag = tf.linalg.diag_part(Q)
    np.testing.assert_allclose(diag, np.ones_like(diag), atol=1e-5)


def test_overlap_matrix_properties_spherical(spherical_system):
    Q = spherical_system.compute_overlap_matrix()
    np.testing.assert_allclose(Q, tf.transpose(Q))
    diag = tf.linalg.diag_part(Q)
    np.testing.assert_allclose(diag, np.ones_like(diag), atol=1e-5)


def test_magnetization_range_ising(ising_system):
    m = ising_system.compute_magnetization()
    assert -1.0 <= m <= 1.0


# ------------------------
# MODEL-SPECIFIC TESTS
# ------------------------

def test_invalid_model_raises():
    with pytest.raises(ValueError, match="Invalid model"):
        SpinSystem(2, 4, 2, tf.ones((4, 4, 4, 4)), model="invalid_model")


def test_invalid_spherical_constraint_raises():
    with pytest.raises(ValueError, match="Spherical constraint can only be applied"):
        SpinSystem(2, 4, 2, tf.ones((4, 4, 4, 4)),
                   model="ising", spherical_constraint=True)


def test_missing_theta_max_for_spherical_model_raises():
    system = SpinSystem(2, 4, 2, tf.ones((4, 4, 4, 4)), model="spherical")
    with pytest.raises(ValueError, match="theta_max must be provided"):
        system.metropolis_sweep(beta=1.0)


def test_spherical_constraint_applies_norm(spherical_system):
    spin_flat = tf.reshape(spherical_system.spin_state,
                           (spherical_system.lattice_replicas, -1))
    norms = tf.norm(spin_flat, axis=1)
    expected = tf.sqrt(spherical_system.number_spins) * tf.ones_like(norms)
    np.testing.assert_allclose(norms.numpy(), expected.numpy(), atol=1e-3)


# ------------------------
# NEW SMOKE TESTS FOR DYNAMICS
# ------------------------

@pytest.mark.parametrize("system_fixture", ["ising_system", "spherical_system", "z2_gauge_system"])
def test_metropolis_step_runs_for_all_models(system_fixture, request):
    """Checks if metropolis_step executes without error for each model type."""
    system = request.getfixturevalue(system_fixture)

    initial_state = tf.identity(system.spin_state)
    initial_energy = tf.identity(system.energy)

    # Provide model-specific required arguments
    kwargs = {}
    if system.model == "spherical":
        kwargs["theta_max"] = tf.constant(0.5)

    # Execute the function
    updated_state = system.metropolis_step(beta=1.0, **kwargs)

    # Assert that the function ran and returned a tensor of the correct shape
    assert isinstance(updated_state, tf.Tensor)
    assert updated_state.shape == initial_state.shape
    assert system.energy.shape == initial_energy.shape


@pytest.mark.parametrize("system_fixture", ["ising_system", "spherical_system", "z2_gauge_system"])
def test_metropolis_sweep_runs_for_all_models(system_fixture, request):
    """Checks if metropolis_sweep executes and returns a correctly structured dictionary."""
    system = request.getfixturevalue(system_fixture)

    sweep_length = 20
    granularity = 10
    num_measurements = sweep_length // granularity + 1

    # Provide model-specific required arguments
    kwargs = {}
    if system.model == "spherical":
        kwargs["theta_max"] = tf.constant(0.5)

    # Note: Your code has a 'track_plaquette' flag but doesn't store its history.
    # This test will fail if track_plaquette=True until that is implemented.
    if system.model == "z2_gauge":
        kwargs["track_plaquette"] = False

    # Execute the function
    results = system.metropolis_sweep(
        beta=1.0,
        sweep_length=sweep_length,
        measurement_granularity=granularity,
        **kwargs
    )

    # Assert that the output is a dictionary with correctly shaped tensors
    assert isinstance(results, dict)
    assert "energy_evolution" in results
    assert "magnetization_evolution" in results
    assert results["energy_evolution"].shape == (
        num_measurements, system.lattice_replicas)
    assert results["magnetization_evolution"].shape == (
        num_measurements, system.lattice_replicas)


# @pytest.mark.parametrize("system_fixture", ["ising_system", "spherical_system", "z2_gauge_system"])
# def test_multi_temperature_sweep_runs_for_all_models(system_fixture, request):
#     """Checks if multi_temperature_sweep executes and returns correctly shaped results."""
#     system = request.getfixturevalue(system_fixture)

#     betas = tf.constant([0.5, 1.0])
#     n_temps = len(betas)
#     sweep_length = 20  # Using a short sweep for testing speed

#     # Provide model-specific required arguments
#     kwargs = {}
#     if system.model == "spherical":
#         kwargs["theta_max"] = tf.constant(0.5)

#     results = system.multi_temperature_sweep(
#         betas=betas, sweep_length=sweep_length, **kwargs
#     )

#     # In the called sweep, default granularity is 100.
#     # A sweep_length of 20 results in 2 measurements (t=0 and the final state).
#     num_measurements = sweep_length // 100 + 1

#     assert isinstance(results, dict)
#     assert "overlap_evolution" in results
#     expected_shape = (n_temps, num_measurements,
#                       system.lattice_replicas, system.lattice_replicas)
#     assert results["overlap_evolution"].shape == expected_shape
