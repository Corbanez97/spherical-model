import tensorflow as tf
import numpy as np
from typing import Union, Optional


class HistoryLogger:
    """Helper to store history only if enabled."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._data = {} if enabled else None
        self._step = 0

    def log(self, **kwargs):
        if not self.enabled:
            return
        self._data[self._step] = {k: (v.numpy() if isinstance(v, tf.Tensor) else v)
                                  for k, v in kwargs.items()}
        self._step += 1

    def get(self):
        return self._data if self.enabled else None


class SpinSystem:
    def __init__(self, N: int,
                 J: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 keep_history: bool = False):
        self.N = N
        self.history = HistoryLogger(keep_history)

        # Coupling matrix J
        if J is None:
            J = tf.random.normal([N, N], stddev=1.0)
            J = (J + tf.transpose(J)) / 2
            J = tf.linalg.set_diag(J, tf.zeros(N))
        else:
            J = tf.convert_to_tensor(J, dtype=tf.float32)
            if J.shape != (N, N):
                raise ValueError(f"J must be shape ({N}, {N})")
        self.J = J

        # Initialize spins
        self.sigma = tf.Variable(self._initialize_spins(), trainable=True)
        self.energy = self.hamiltonian(self.sigma, self.J).numpy()

    def _initialize_spins(self) -> tf.Tensor:
        sigma = tf.random.normal([self.N])
        sigma = self._apply_spherical_constraint(sigma)
        # magnetization = tf.reduce_mean(sigma)
        # self.history.log(sigma=sigma)
        return sigma

    def _apply_spherical_constraint(self, sigma: tf.Tensor) -> tf.Tensor:
        return tf.sqrt(tf.cast(self.N, tf.float32)) * tf.math.l2_normalize(sigma)

    def hamiltonian(self, sigma: tf.Tensor, J: tf.Tensor) -> tf.Tensor:
        energy = -tf.einsum('i,ij,j->', sigma, J, sigma)
        magnetization = tf.reduce_mean(sigma)
        self.history.log(sigma=sigma, energy=energy,
                         magnetization=magnetization)
        return energy

    def _rotate_spins(self, i: int, j: int, theta: float):
        cos_t, sin_t = tf.cos(theta), tf.sin(theta)
        sigma_i, sigma_j = self.sigma[i], self.sigma[j]

        new_i = cos_t * sigma_i - sin_t * sigma_j
        new_j = sin_t * sigma_i + cos_t * sigma_j

        return new_i, new_j

    def metropolis_step(self, beta: float, theta_max: float = 0.1) -> bool:
        i, j = np.random.choice(self.N, size=2, replace=False)
        theta = tf.random.uniform([], -theta_max, theta_max)

        new_i, new_j = self._rotate_spins(i, j, theta)

        # Candidate new sigma
        next_sigma = tf.tensor_scatter_nd_update(
            self.sigma,
            indices=[[i], [j]],
            updates=[new_i, new_j],
        )

        next_energy = self.hamiltonian(next_sigma, self.J).numpy()
        dE = next_energy - self.energy

        if dE < 0 or tf.random.uniform([]) < tf.exp(-beta * dE):
            # Accept move
            self.sigma.assign(next_sigma)
            self.energy = next_energy
            magnetization = tf.reduce_mean(self.sigma)
            self.history.log(accepted=True, dE=dE,
                             sigma=self.sigma, energy=self.energy, magnetization=magnetization)
            return True

        # Reject → state unchanged
        self.history.log(accepted=False, dE=dE, energy=self.energy)
        return False

    def metropolis_sweep(self, beta: float, steps: Optional[int] = None, theta_max: float = 0.1):
        if steps is None:
            steps = self.N

        accepted = sum(self.metropolis_step(beta, theta_max)
                       for _ in range(steps))
        acceptance_rate = accepted / steps
        print(
            f"Metropolis sweep: {accepted}/{steps} accepted ({acceptance_rate:.2%})")
        return acceptance_rate

    def optimize_metropolis(self, beta: float, sweeps: int = 100, theta_max: float = 0.1):
        for sweep in range(sweeps):
            acceptance_rate = self.metropolis_sweep(beta, self.N, theta_max)
            energy = self.hamiltonian()

            if sweep % 10 == 0:
                print(f"Sweep {sweep}: Energy = {energy.numpy():.4f}, "
                      f"Acceptance = {acceptance_rate:.2%}")
