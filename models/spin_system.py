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
                 ising: bool = False,
                 bias: float = 0.5):              # bias strength
        self.D = D
        self.L = L
        self.shape = [L] * D
        self.N = L**D
        self.history = HistoryLogger(keep_history)
        self.spherical = spherical
        self.ising = ising
        self.bias = bias  # store bias

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
            # Probability of spin being +1 is shifted by bias
            # bias > 0 favors +1, bias < 0 favors -1
            p_up = 0.5 + 0.5 * tf.tanh(self.bias)  # keeps it in [0,1]
            sigma = tf.cast(tf.random.uniform(self.shape) < p_up, tf.float32)
            sigma = 2 * sigma - 1
        else:
            # Continuous spins with mean bias
            sigma = tf.random.normal(self.shape, mean=self.bias, stddev=1.0)
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
        return -0.5*tf.einsum(einsum_str, sigma, self.J, sigma)  # O(N^{2^N})

    def _compute_energy_delta(self, coords: list, new_vals: tf.Tensor) -> tf.Tensor:
        # Flatten spins for indexing
        sigma_flat = tf.reshape(self.sigma, [self.N])
        J_flat = tf.reshape(self.J, (self.N, self.N))

        # Convert multi-d coords into flat indices
        flat_indices = [np.ravel_multi_index(c, self.shape) for c in coords]

        dE = 0.0
        for idx, new_val in zip(flat_indices, tf.unstack(new_vals)):
            old_val = sigma_flat[idx]
            delta_sigma = new_val - old_val

            # Contribution from coupling with all other spins
            interaction = tf.reduce_sum(J_flat[idx, :] * sigma_flat)

            dE += -delta_sigma * interaction - 0.5 * \
                J_flat[idx, idx] * (new_val**2 - old_val**2)

        return dE

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

        # next_energy = self._compute_energy(next_sigma)
        # dE = next_energy - self.energy

        dE = self._compute_energy_delta(
            coords=[coords1, coords2] if not self.ising else [coords],
            new_vals=tf.stack(
                [new_i, new_j]) if not self.ising else tf.stack([new_val])
        )

        if dE < 0 or tf.random.uniform([]) < tf.exp(-beta * dE):
            self.sigma.assign(next_sigma)
            self.energy = self.energy + dE
            self._log_state(self.sigma, energy=self.energy,
                            accepted=True, dE=dE.numpy())
            return True

        self.history.log(accepted=False, dE=dE.numpy())
        return False

    def metropolis_sweep(self, beta: float, steps: Optional[int] = None, accepted_steps: Optional[int] = None, theta_max: float = 0.1) -> float:
        if steps is None and accepted_steps is None:
            steps = self.L

        accepted = 0
        attempted = 0

        if steps is not None:
            for _ in range(steps):
                accepted += self.metropolis_step(beta, theta_max)
            attempted = steps

        else:
            while accepted < accepted_steps:
                accepted += self.metropolis_step(beta, theta_max)
                attempted += 1

        acceptance_rate = accepted / attempted if attempted > 0 else 0.0
        print(
            f"Metropolis sweep: {accepted}/{attempted} accepted ({acceptance_rate:.2%})"
        )
        return acceptance_rate

    def optimize_metropolis(self, beta: float, sweeps: int = 100,
                            theta_max: float = 0.1) -> None:
        for sweep in range(sweeps):
            acceptance_rate = self.metropolis_sweep(beta, self.L, theta_max)

            if sweep % 10 == 0:
                print(f"Sweep {sweep}: Energy = {self.energy.numpy():.4f}, "
                      f"Acceptance = {acceptance_rate:.2%}")
