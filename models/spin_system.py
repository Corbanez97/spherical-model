import tensorflow as tf
import numpy as np
from typing import Union, Optional, Dict, Any


class HistoryLogger:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._data: Dict[int, Dict[str, Any]] = {} if enabled else None
        self._step = 0

    def log(self, **kwargs) -> None:
        if not self.enabled:
            return
        self._data[self._step] = {
            k: (v.numpy() if isinstance(v, tf.Tensor) else v)
            for k, v in kwargs.items()
        }
        self._step += 1

    def get(self) -> Optional[Dict[int, Dict[str, Any]]]:
        return self._data if self.enabled else None


class SpinSystem:
    def __init__(self,
                 D: int,
                 L: int,
                 J: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 keep_history: bool = False,
                 spherical: bool = True,          # enable/disable spherical constraint
                 # use Ising spins (±1) instead of continuous
                 ising: bool = False):
        self.D = D
        self.L = L
        self.shape = [L] * D
        self.N = L**D
        self.history = HistoryLogger(keep_history)
        self.spherical = spherical
        self.ising = ising

        if J is None:
            # J ~ [L, L, ..., L] (2*D times)
            J = tf.random.normal(self.shape + self.shape, stddev=1.0)
            axes = list(range(2*D))
            perm = axes[D:] + axes[:D]   # swap halves
            J = 0.5 * (J + tf.transpose(J, perm=perm))
            mask = 1.0 - tf.eye(self.N)
            J = tf.reshape(J, (self.N, self.N)) * mask
            J = tf.reshape(J, self.shape + self.shape)
        else:
            J = tf.convert_to_tensor(J, dtype=tf.float32)
            if tuple(J.shape) != tuple(self.shape + self.shape):
                raise ValueError(
                    f"J must be shape {self.shape + self.shape}, got {J.shape}"
                )
        self.J = J

        self.sigma = tf.Variable(self._initialize_spins(), trainable=True)
        self.energy = self._compute_energy(self.sigma)
        self._log_initial_state()

    def _initialize_spins(self) -> tf.Tensor:
        if self.ising:
            # Random ±1
            sigma = tf.random.uniform(
                self.shape, minval=0, maxval=2, dtype=tf.int32)
            sigma = tf.cast(2 * sigma - 1, tf.float32)
        else:
            # Continuous spins
            sigma = tf.random.normal(self.shape)
            if self.spherical:
                sigma = self._apply_spherical_constraint(sigma)
        return sigma

    def _apply_spherical_constraint(self, sigma: tf.Tensor) -> tf.Tensor:
        if not self.spherical:
            return sigma
        return tf.sqrt(tf.cast(self.N, tf.float32)) * tf.math.l2_normalize(sigma)

    def _compute_energy(self, sigma: tf.Tensor) -> tf.Tensor:
        # A very hacky way to perform the summation for arbitrary dimensions
        #   sigma_{i1,...,iD} J_{i1,...,iD,j1,...,jD} sigma_{j1,...,jD}
        ndim = sigma.numpy().ndim
        letters = "abcdefghijklmnopqrstuvwxyz"
        assert 2*ndim <= len(letters), "Too many dimensions!"

        left = letters[:ndim]
        right = letters[ndim:2*ndim]
        einsum_str = f"{left},{left+right},{right}->"
        return -tf.einsum(einsum_str, sigma, self.J, sigma)

    def _compute_magnetization(self, sigma: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(sigma)

    def _log_state(self, sigma: tf.Tensor, energy: tf.Tensor,
                   accepted: Optional[bool] = None, dE: Optional[float] = None) -> None:
        log_data = {
            'sigma': tf.identity(sigma),
            'energy': energy,
            'magnetization': self._compute_magnetization(sigma)
        }
        if accepted is not None:
            log_data['accepted'] = accepted
        if dE is not None:
            log_data['dE'] = dE
        self.history.log(**log_data)

    def _log_initial_state(self) -> None:
        """Log initial state without redundant energy computation"""
        self._log_state(self.sigma, self.energy)

    def metropolis_step(self, beta: float, theta_max: float = 0.1) -> bool:
        if self.ising:
            # ---- Ising flip ----
            coords = [np.random.randint(0, self.L) for _ in range(self.D)]

            sigma_val = tf.gather_nd(self.sigma, [coords])
            new_val = -sigma_val  # flip

            next_sigma = tf.tensor_scatter_nd_update(
                self.sigma,
                indices=[coords],
                updates=new_val
            )
        else:
            # ---- Continuous rotation ----
            coords1 = [np.random.randint(0, self.L) for _ in range(self.D)]
            coords2 = [np.random.randint(0, self.L) for _ in range(self.D)]
            while coords1 == coords2:
                coords2 = [np.random.randint(0, self.L) for _ in range(self.D)]

            theta = tf.random.uniform([], -theta_max, theta_max)
            cos_t, sin_t = tf.cos(theta), tf.sin(theta)

            sigma_i = tf.gather_nd(self.sigma, [coords1])
            sigma_j = tf.gather_nd(self.sigma, [coords2])

            new_i = cos_t * sigma_i - sin_t * sigma_j
            new_j = sin_t * sigma_i + cos_t * sigma_j

            next_sigma = tf.tensor_scatter_nd_update(
                self.sigma,
                indices=[coords1],
                updates=new_i
            )
            next_sigma = tf.tensor_scatter_nd_update(
                next_sigma,
                indices=[coords2],
                updates=new_j
            )

        # --- Metropolis acceptance ---
        if self.spherical and not self.ising:
            next_sigma = self._apply_spherical_constraint(next_sigma)

        next_energy = self._compute_energy(next_sigma)
        dE = next_energy - self.energy

        if dE < 0 or tf.random.uniform([]) < tf.exp(-beta * dE):
            self.sigma.assign(next_sigma)
            self.energy = next_energy
            self._log_state(self.sigma, self.energy,
                            accepted=True, dE=dE.numpy())
            return True

        self.history.log(accepted=False, dE=dE.numpy(), energy=self.energy)
        return False

    def metropolis_sweep(self, beta: float, steps: Optional[int] = None,
                         theta_max: float = 0.1) -> float:
        if steps is None:
            steps = self.L

        accepted = sum(self.metropolis_step(beta, theta_max)
                       for _ in range(steps))
        acceptance_rate = accepted / steps
        print(
            f"Metropolis sweep: {accepted}/{steps} accepted ({acceptance_rate:.2%})")
        return acceptance_rate

    def optimize_metropolis(self, beta: float, sweeps: int = 100,
                            theta_max: float = 0.1) -> None:
        for sweep in range(sweeps):
            acceptance_rate = self.metropolis_sweep(beta, self.L, theta_max)

            if sweep % 10 == 0:
                print(f"Sweep {sweep}: Energy = {self.energy.numpy():.4f}, "
                      f"Acceptance = {acceptance_rate:.2%}")
