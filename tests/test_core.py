import pytest
import tensorflow as tf
import numpy as np
from spin_system.core import SpinSystem


# ------------------------
# FIXTURES
# ------------------------
@pytest.fixture
def ising_system():
    L = 4
    replicas = 2
    shape = (L, L)
    J = tf.ones(shape + shape) / (L * L)  # uniform coupling

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


def test_magnetization_range_spherical(spherical_system):
    m = spherical_system.compute_magnetization()
    assert -1.0 <= m <= 1.0


# ------------------------
# DYNAMICS TESTS
# ------------------------
def test_metropolis_step_updates_state_ising(ising_system):
    tf.random.set_seed(0)
    old_state = tf.identity(ising_system.spin_state)
    ising_system.metropolis_step(beta=0.1)
    new_state = ising_system.spin_state
    assert not tf.reduce_all(tf.equal(old_state, new_state))


def test_metropolis_step_updates_state_spherical(spherical_system):
    tf.random.set_seed(0)
    old_state = tf.identity(spherical_system.spin_state)
    spherical_system.metropolis_step(beta=0.1, theta_max=1.0)
    new_state = spherical_system.spin_state
    assert not tf.reduce_all(tf.equal(old_state, new_state))


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
