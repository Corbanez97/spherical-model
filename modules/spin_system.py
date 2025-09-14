import tensorflow as tf
import numpy as np
from typing import Union, Optional, Dict, Any, List


class HistoryLogger:
    """A simple class to log simulation history.

    This logger stores data from each step of a simulation in a dictionary.
    It can be disabled to save memory.

    Attributes:
        enabled (bool): If True, logging is active.
        _data (Optional[Dict[int, Dict[str, Any]]]): A dictionary where keys are
            simulation steps and values are dictionaries of logged data.
        _step (int): The current simulation step counter.
    """

    def __init__(self, enabled: bool) -> None:
        """Initializes the HistoryLogger.

        Args:
            enabled (bool): A flag to enable or disable logging.
        """
        self.enabled = enabled
        self._data: Optional[Dict[int, Dict[str, Any]]
                             ] = {} if enabled else None
        self._step = 0

    def log(self, **kwargs: Any) -> None:
        """Logs key-value data for the current step.

        If the logger is disabled, this method does nothing. It automatically
        converts TensorFlow tensors to NumPy arrays for easier handling.

        Args:
            **kwargs: Arbitrary keyword arguments representing the data to log.
        """
        if not self.enabled:
            return
        # The check 'self._data is not None' is implicitly handled by 'self.enabled'
        self._data[self._step] = {
            k: (v.numpy() if isinstance(v, tf.Tensor) else v)
            for k, v in kwargs.items()
        }
        self._step += 1

    def get(self) -> Optional[Dict[int, Dict[str, Any]]]:
        """Returns the complete history data.

        Returns:
            An optional dictionary containing the entire log, or None if disabled.
        """
        return self._data if self.enabled else None

# ------------------------------------------------------------------------------


class SpinSystem:
    """Simulates a generic spin system (spin glass) model.

    This class can handle D-dimensional lattices of continuous (vector) or
    discrete (Ising) spins. It supports a spherical constraint, which forces
    the sum of squares of all spins to be equal to the total number of spins N.
    The simulation primarily uses the Metropolis-Hastings algorithm.

    Attributes:
        D (int): The number of dimensions of the lattice.
        L (int): The size of the lattice in each dimension.
        N (int): The total number of spins (L**D).
        J (tf.Tensor): The coupling matrix of shape [L,...,L,L,...,L] (2*D dims).
        sigma (tf.Variable): The tensor of spin states of shape [L,...,L].
        energy (tf.Tensor): The current energy of the system.
        history (HistoryLogger): A logger for tracking simulation states.
        spherical (bool): Flag for the spherical constraint on continuous spins.
        ising (bool): Flag for using discrete Ising spins (±1).
        bias (float): A bias term favoring positive spin values.
    """

    def __init__(self,
                 D: int,
                 L: int,
                 J: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 keep_history: bool = False,
                 spherical: bool = True,
                 ising: bool = False,
                 bias: float = 0.5) -> None:
        """Initializes the SpinSystem.

        Args:
            D (int): The number of dimensions for the spin lattice.
            L (int): The size of the lattice along each dimension.
            J (Optional[Union[tf.Tensor, np.ndarray]]): The interaction tensor.
                If None, a random symmetric tensor from a Gaussian distribution
                is generated. Defaults to None.
            keep_history (bool): If True, a HistoryLogger is enabled. Defaults to False.
            spherical (bool): If True, applies the spherical constraint to continuous
                spins. Defaults to True.
            ising (bool): If True, uses discrete Ising spins (±1). Defaults to False.
            bias (float): A bias term that favors positive spin values.
                For Ising spins, it shifts the probability of initializing to +1.
                For continuous spins, it sets the mean of the initial distribution.
                Defaults to 0.5.

        Raises:
            ValueError: If the provided J tensor has an incorrect shape.
        """
        self.D = D
        self.L = L
        self.shape = [L] * D
        self.N = L**D
        self.history = HistoryLogger(keep_history)
        self.spherical = spherical
        self.ising = ising
        self.bias = bias

        if J is None:
            # J ~ [L, L, ..., L] (2*D times)
            J = tf.random.normal(self.shape + self.shape, stddev=1.0)
            axes = list(range(2*D))
            perm = axes[D:] + axes[:D]  # swap halves
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
        """Generates the initial spin configuration based on model parameters.

        Returns:
            A TensorFlow tensor representing the initial state of the spins.
        """
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
        """Normalizes spins to satisfy the spherical constraint sum(s_i^2) = N.

        Args:
            sigma (tf.Tensor): The spin configuration tensor.

        Returns:
            The normalized spin tensor. Returns the original tensor if the
            spherical constraint is disabled.
        """
        if not self.spherical:
            return sigma
        return tf.sqrt(tf.cast(self.N, tf.float32)) * tf.math.l2_normalize(sigma)

    def _compute_energy(self, sigma: tf.Tensor) -> tf.Tensor:
        """Computes the total energy of the system via Hamiltonian H = -0.5 * sum(s_i J_ij s_j).

        This method uses Einstein summation to handle arbitrary dimensions. The
        complexity is O(N^2), where N is the total number of spins.

        Args:
            sigma (tf.Tensor): The spin configuration tensor.

        Returns:
            A scalar TensorFlow tensor representing the total energy.
        """
        ndim = sigma.numpy().ndim
        letters = "abcdefghijklmnopqrstuvwxyz"
        assert 2*ndim <= len(letters), "Too many dimensions!"

        left = letters[:ndim]
        right = letters[ndim:2*ndim]
        einsum_str = f"{left},{left+right},{right}->"
        return -0.5*tf.einsum(einsum_str, sigma, self.J, sigma)

    def _compute_energy_delta(self, coords: List[List[int]], new_vals: tf.Tensor) -> tf.Tensor:
        """Computes the change in energy from updating a few spins.

        This is much faster than recomputing the total energy when only a few
        spins change.

        Args:
            coords (List[List[int]]): A list of coordinates for the updated spins.
            new_vals (tf.Tensor): A 1D tensor of the new spin values at `coords`.

        Returns:
            A scalar tensor for the change in energy (dE).
        """
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

            # The diagonal J_ii is zero by construction, but we keep this term
            # for generality in case a model with J_ii != 0 is used.
            dE += -delta_sigma * interaction - 0.5 * \
                J_flat[idx, idx] * (new_val**2 - old_val**2)

        return dE

    def _compute_magnetization(self, sigma: tf.Tensor) -> tf.Tensor:
        """Computes the average magnetization of the system.

        Args:
            sigma (tf.Tensor): The spin configuration tensor.

        Returns:
            A scalar tensor for the average magnetization.
        """
        return tf.reduce_mean(sigma)

    def _log_state(self, sigma: tf.Tensor, energy: tf.Tensor,
                   accepted: Optional[bool] = None, dE: Optional[float] = None) -> None:
        """Logs the current state of the system.

        Args:
            sigma (tf.Tensor): The current spin configuration.
            energy (tf.Tensor): The current total energy.
            accepted (Optional[bool]): Whether the last Metropolis step was accepted.
            dE (Optional[float]): The energy change from the last step.
        """
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
        """Logs the initial state after initialization."""
        self._log_state(self.sigma, self.energy)

    def metropolis_step(self, beta: float, theta_max: float = 0.1) -> bool:
        """Performs a single Metropolis-Hastings step.

        For Ising spins, this involves flipping a single randomly chosen spin.
        For continuous spins, it involves rotating two randomly chosen spins
        in their shared plane by a small random angle (a Givens rotation).

        Args:
            beta (float): The inverse temperature ($1/k_B T$).
            theta_max (float): The maximum rotation angle for continuous spins.

        Returns:
            A boolean indicating whether the proposed move was accepted.
        """
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

        if self.spherical and not self.ising:
            next_sigma = self._apply_spherical_constraint(next_sigma)

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

    def metropolis_sweep(self, beta: float, steps: Optional[int] = None,
                         accepted_steps: Optional[int] = None, theta_max: float = 0.1) -> float:
        """Runs multiple Metropolis steps.

        The sweep can run for a fixed number of total steps or until a target
        number of accepted steps is reached. One of `steps` or `accepted_steps` must be provided.

        Args:
            beta (float): The inverse temperature ($1/k_B T$).
            steps (Optional[int]): The total number of steps to attempt. Defaults to self.L.
            accepted_steps (Optional[int]): The target number of accepted steps.
            theta_max (float): Max rotation angle for continuous spins.

        Returns:
            The acceptance rate for the sweep.
        """
        if steps is None and accepted_steps is None:
            steps = self.L

        accepted = 0
        attempted = 0

        if steps is not None:
            for _ in range(steps):
                accepted += self.metropolis_step(beta, theta_max)
            attempted = steps

        else:  # accepted_steps is not None
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
        """Runs a full simulation for a given number of sweeps.

        Note: The name is a misnomer; this method runs a simulation, it does not
        optimize the parameters of the Metropolis algorithm itself.

        Args:
            beta (float): The inverse temperature ($1/k_B T$).
            sweeps (int): The number of sweeps to perform.
            theta_max (float): Max rotation angle for continuous spins.
        """
        for sweep in range(sweeps):
            # A "sweep" is often defined as N individual steps, but here it's L.
            # This is a detail of the implementation.
            acceptance_rate = self.metropolis_sweep(
                beta, self.L, theta_max=theta_max)

            if sweep % 10 == 0:
                print(f"Sweep {sweep}: Energy = {self.energy.numpy():.4f}, "
                      f"Acceptance = {acceptance_rate:.2%}")
