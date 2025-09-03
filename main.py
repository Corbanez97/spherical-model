"""
Spin System Hamiltonian Calculator with Spherical Constraint

This script computes the Hamiltonian for a system of N spins with arbitrary
pairwise interactions J_{ij} and a spherical constraint sum(\sigma_i^2) = N.

The Hamiltonian is defined as:
    H = \sigma_{i,j} J_{ij} \sigma_i \sigma_j

The spherical constraint is enforced by normalizing the spin vector to have
L2 norm equal to √N.

TensorFlow is used for efficient computation, especially beneficial for:
1. Large systems (GPU acceleration)
2. Automatic differentiation for optimization
3. Easy integration with machine learning approaches
"""

import tensorflow as tf
import numpy as np
from typing import Union, Optional


class SpinSystem:
    """
    A class representing a system of spins with arbitrary interactions.

    Attributes:
        N (int): Number of spins in the system
        J (tf.Tensor): Interaction matrix of shape (N, N)
        sigma (tf.Variable): Spin configuration vector of shape (N,)
    """

    def __init__(self, N: int, J: Optional[Union[tf.Tensor, np.ndarray]] = None):
        """
        Initialize the spin system.

        Args:
            N: Number of spins in the system
            J: Interaction matrix (N x N). If None, creates a random symmetric matrix.
        """
        self.N = N

        # Create or validate the interaction matrix
        if J is None:
            # Create a random symmetric matrix with zero diagonal
            J = tf.random.normal([N, N], stddev=1.0)
            J = (J + tf.transpose(J)) / 2  # Make symmetric
            J = tf.linalg.set_diag(J, tf.zeros(N))  # No self-interactions
        else:
            # Convert to tensor if needed and validate shape
            J = tf.convert_to_tensor(J, dtype=tf.float32)
            if J.shape != [N, N]:
                raise ValueError(f"J must be of shape [{N}, {N}]")

        self.J = J

        # Initialize spins with spherical constraint
        self.sigma = tf.Variable(self._initialize_spins(), trainable=True)

    def _initialize_spins(self) -> tf.Tensor:
        """Initialize spins with spherical constraint."""
        sigma = tf.random.normal([self.N])
        return self._apply_spherical_constraint(sigma)

    def _apply_spherical_constraint(self, sigma: tf.Tensor) -> tf.Tensor:
        """
        Apply spherical constraint to spin vector.

        Args:
            sigma: Spin vector to normalize

        Returns:
            Normalized spin vector satisfying \sum\sigma_i^2 = N
        """
        return tf.sqrt(tf.cast(self.N, tf.float32)) * tf.math.l2_normalize(sigma)

    def hamiltonian(self) -> tf.Tensor:
        """
        Compute the Hamiltonian H = \sigma_{i,j} J_{ij} \sigma_i \sigma_j.

        Returns:
            Scalar value of the Hamiltonian

        Mathematical formulation:
            The Hamiltonian can be written in matrix form as:
                H = \sigma^T J \sigma
            where \sigma is the vector of spins and J is the interaction matrix.

        TensorFlow implementation:
            We compute this using Einstein summation notation:
                H = tf.einsum('i,ij,j->', sigma, J, sigma)
            which is equivalent to:
                H = \sigma_i \sigma_j \sigma_i * J_{ij} * \sigma_j
        """
        return -tf.einsum('i,ij,j->', self.sigma, self.J, self.sigma)

    def optimize_energy(self, learning_rate: float = 0.01, steps: int = 100) -> None:
        """
        Optimize the spin configuration to minimize energy using gradient descent.

        Args:
            learning_rate: Step size for gradient descent
            steps: Number of optimization steps
        """
        optimizer = tf.optimizers.SGD(learning_rate)

        for step in range(steps):
            with tf.GradientTape() as tape:
                energy = self.hamiltonian()

            # Compute gradients
            gradients = tape.gradient(energy, [self.sigma])

            # Apply gradient descent
            optimizer.apply_gradients(zip(gradients, [self.sigma]))

            # Reapply spherical constraint after update
            self.sigma.assign(self._apply_spherical_constraint(self.sigma))

            if step % 20 == 0:
                print(f"Step {step}: Energy = {energy.numpy():.4f}")


def main():
    """Example usage of the SpinSystem class."""
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # System parameters
    N = 100  # Number of spins

    print("=== Spin System Hamiltonian Calculator ===")
    print(f"Number of spins: {N}")
    print("Spherical constraint: \sum\sigma_i^2 = N")
    print()

    # Create spin system with random interactions
    system = SpinSystem(N)

    # Compute initial energy
    initial_energy = system.hamiltonian()
    print(f"Initial energy: {initial_energy.numpy():.4f}")

    # Verify spherical constraint
    constraint_check = tf.reduce_sum(system.sigma ** 2).numpy()
    print(
        f"Spherical constraint check (should be {N}): {constraint_check:.2f}")
    print()

    # Optimize the energy
    print("Optimizing energy...")
    system.optimize_energy(learning_rate=0.1, steps=100)

    # Final energy
    final_energy = system.hamiltonian()
    print(f"Final energy: {final_energy.numpy():.4f}")

    # Verify constraint is still satisfied
    constraint_check = tf.reduce_sum(system.sigma ** 2).numpy()
    print(f"Final constraint check (should be {N}): {constraint_check:.2f}")


if __name__ == "__main__":
    main()
