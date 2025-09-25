import tensorflow as tf
import numpy as np
from typing import Union, Optional, Callable, Tuple, Literal, Dict


class SpinSystem(tf.Module):
    def __init__(
        self,
        lattice_dim: int,
        lattice_length: int,
        interaction_matrix: Union[tf.Tensor, np.ndarray],
        initial_spin_state: Optional[Union[tf.Tensor, np.ndarray]] = None,
        external_field: Optional[Union[tf.Tensor, np.ndarray]] = None,
        model: Literal["ising", "spherical", "z2_gauge"] = "ising",
        spherical_constraint: bool = False,
        initial_magnetization: float = 0.5,
    ) -> None:

        assert not (model != "spherical" and spherical_constraint), \
            "Spherical constraint can only be applied to spherical model!"

        self.lattice_dim = lattice_dim
        self.lattice_length = lattice_length
        self.shape = [lattice_length] * lattice_dim
        self.number_spins = tf.cast(lattice_length ** lattice_dim, tf.float32)
        self.model = model
        self.spherical_contraint = spherical_constraint
        self.initial_magnetization = initial_magnetization

        self.interaction_matrix = self._validate_tensor_shape(
            interaction_matrix,
            expected_shape=self.shape + self.shape,
            name="Interaction matrix",
        )

        self.external_field = self._validate_tensor_shape(
            external_field,
            expected_shape=self.shape,
            name="External field",
            allow_none=True,
            default=tf.zeros(self.shape, dtype=tf.float32),
        )

        self.spin_state = self._validate_tensor_shape(
            initial_spin_state,
            expected_shape=self.shape,
            name="Initial spin state",
            allow_none=True,
            default=lambda: tf.Variable(
                self._initialize_spins_state(), trainable=True
            ),
        )

        self.energy = self.compute_pairwise_energy()

    def _validate_tensor_shape(
        self,
        tensor: Optional[tf.Tensor],
        expected_shape: tuple[int, ...],
        name: str,
        allow_none: bool = False,
        default: Optional[Union[tf.Tensor, Callable[[], tf.Tensor]]] = None,
    ) -> tf.Tensor:
        """
        Convert input to tf.Tensor and validate its shape.
        If None is allowed, returns a default if provided.
        """
        if tensor is None:
            if allow_none:
                if callable(default):
                    return default()
                elif default is not None:
                    return default
                else:
                    return None
            else:
                raise ValueError(f"{name} cannot be None")

        tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} must be shape {expected_shape}, got {tensor.shape}"
            )
        return tensor

    @tf.function
    def _apply_spherical_constraint(self, spin_state: tf.Tensor) -> tf.Tensor:
        return tf.sqrt(self.number_spins) * tf.math.l2_normalize(spin_state)

    def _initialize_spins_state(self) -> tf.Tensor:
        if self.model == "ising":
            p_up = 0.5 + 0.5 * tf.tanh(self.initial_magnetization)
            spin_state = tf.cast(tf.random.uniform(
                self.shape) < p_up, tf.float32)
            spin_state = 2 * spin_state - 1
        elif self.model == "spherical":
            # TODO: We could implement an argument that enables a different initial distribution for spins
            spin_state = tf.random.normal(
                self.shape, mean=self.initial_magnetization, stddev=1.0)
            if self.spherical_contraint:
                spin_state = self._apply_spherical_constraint(spin_state)
        elif self.model == "z2_gauge":
            raise NotImplementedError("Z2 gauge model not implemented yet")
        return spin_state

    # TODO: Compute pairwise energy delta

    @tf.function
    def compute_pairwise_energy(
        self,
        spin_state: Optional[tf.Tensor] = None,
        interaction_matrix: Optional[tf.Tensor] = None,
        external_field: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:

        assert self.model != "z2_gauge", "Can not compute pairwise energy for Z2 gauge model"

        if spin_state is None:
            spin_state = self.spin_state
        if interaction_matrix is None:
            interaction_matrix = self.interaction_matrix
        if external_field is None:
            external_field = self.external_field

        spin_state_flat = tf.reshape(spin_state, (1, self.number_spins))
        interaction_matrix_flat = tf.reshape(
            interaction_matrix, (self.number_spins, self.number_spins))
        external_field_flat = tf.reshape(
            external_field, (1, self.number_spins))

        pairwise = -0.5 * \
            (spin_state_flat @ interaction_matrix_flat @
             tf.transpose(spin_state_flat))
        field_term = - external_field_flat @ tf.transpose(spin_state_flat)

        return tf.squeeze(pairwise + field_term)

    # TODO: Compute the magnetization susceptibility as the variance of the magnetization. I need to first use a populational method to compute the mag

    @tf.function
    def compute_magnetization(self, spin_state: Optional[tf.Tensor] = None) -> tf.Tensor:
        if spin_state is None:
            spin_state = self.spin_state
        return tf.reduce_mean(spin_state)

    @tf.function
    def flip_spins(self, num_flips: tf.Tensor, spin_state: Optional[tf.Tensor] = None, spin_flat: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Flip `num_flips` random spins in Ising model."""
        both_none = (spin_state is None) and (spin_flat is None)
        exactly_one = (spin_state is not None) ^ (spin_flat is not None)

        tf.debugging.assert_equal(both_none or exactly_one, True,
                                  message="Either provide no arguments or exactly one of spin_state or spin_flat")

        if spin_state is not None:
            spin_flat = tf.reshape(spin_state, [-1])
        elif spin_flat is None:
            spin_flat = tf.reshape(self.spin_state, [-1])

        # num_flips = tf.cast(num_flips, tf.int32)
        idx = tf.random.shuffle(tf.range(self.number_spins, dtype=tf.int32))[
            :num_flips]
        updates = -tf.gather(spin_flat, idx)
        return tf.tensor_scatter_nd_update(spin_flat,
                                           indices=tf.expand_dims(idx, 1),
                                           updates=updates)

    @tf.function
    def rotate_spins(self, num_pairs: tf.Tensor, theta_max: tf.Tensor,
                     spin_state: Optional[tf.Tensor] = None, spin_flat: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Rotate `num_pairs` pairs of spins by random angles in [-theta_max, theta_max]."""

        both_none = (spin_state is None) and (spin_flat is None)
        exactly_one = (spin_state is not None) ^ (
            spin_flat is not None)  # XOR operation

        tf.debugging.assert_equal(both_none or exactly_one, True,
                                  message="Either provide no arguments or exactly one of spin_state or spin_flat")

        if spin_state is not None:
            spin_flat = tf.reshape(spin_state, [-1])
        elif spin_flat is None:
            spin_flat = tf.reshape(self.spin_state, [-1])
        # num_pairs = tf.cast(num_pairs, tf.int32)
        idx = tf.random.shuffle(tf.range(self.number_spins, dtype=tf.int32))[
            : 2 * num_pairs]
        idx1, idx2 = idx[:num_pairs], idx[num_pairs:]

        sigma_i = tf.gather(spin_flat, idx1)
        sigma_j = tf.gather(spin_flat, idx2)

        theta = tf.random.uniform([num_pairs], -theta_max, theta_max)
        cos_t, sin_t = tf.cos(theta), tf.sin(theta)

        new_i = cos_t * sigma_i - sin_t * sigma_j
        new_j = sin_t * sigma_i + cos_t * sigma_j

        updates = tf.concat([new_i, new_j], axis=0)
        indices = tf.concat([tf.expand_dims(idx1, 1),
                            tf.expand_dims(idx2, 1)], axis=0)

        return tf.tensor_scatter_nd_update(spin_flat, indices, updates)

    @tf.function
    def _disturb_state(self, num_disturb: Optional[tf.Tensor], theta_max: Optional[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:

        if self.model == "ising":
            new_spin_flat = self.flip_spins(num_disturb)
        elif self.model == "spherical":
            new_spin_flat = self.rotate_spins(num_disturb, theta_max)
        elif self.model == "z2_gauge":
            raise NotImplementedError("")

        next_spin_state = tf.reshape(new_spin_flat, self.spin_state.shape)

        if self.spherical_contraint:
            next_spin_state = self._apply_spherical_constraint(next_spin_state)

        energy_delta = self.compute_pairwise_energy(
            next_spin_state) - self.compute_pairwise_energy()

        return next_spin_state, energy_delta

    @tf.function
    def metropolis_step(self, beta: float, num_disturb: Optional[tf.Tensor] = 1, theta_max: Optional[tf.Tensor] = None) -> tf.Tensor:
        next_spin_state, energy_delta = self._disturb_state(
            num_disturb=num_disturb,
            theta_max=theta_max
        )

        accept = tf.logical_or(
            energy_delta < 0.0,
            tf.random.uniform([]) < tf.exp(-beta * energy_delta)
        )
        new_spin_state = tf.where(accept, next_spin_state, self.spin_state)

        self.spin_state.assign(new_spin_state)

        return new_spin_state

    @tf.function
    def metropolis_sweep(
        self,
        beta: float,
        num_disturb: Optional[tf.Tensor] = 1,
        theta_max: Optional[tf.Tensor] = None,
        sweep_length: int = 100,
        measurement_granularity: int = 100,
        track_spins: bool = True,
        track_energy: bool = True,
        track_magnetization: bool = True,
    ) -> Dict:
        def make_array(track, size):
            # This is bad because the code tries to access a tensor even if I don't want to track it...
            return tf.TensorArray(dtype=tf.float32, size=size) if track else tf.TensorArray(dtype=tf.float32, size=0)

        spin_evolution = make_array(
            track_spins, int(round(sweep_length/measurement_granularity)) + 1)
        energy_evolution = make_array(
            track_energy, int(round(sweep_length/measurement_granularity)) + 1)
        magnetization_evolution = make_array(
            track_magnetization, int(round(sweep_length/measurement_granularity)) + 1)

        if track_spins:
            spin_evolution = spin_evolution.write(0, self.spin_state)
        if track_energy:
            energy_evolution = energy_evolution.write(
                0, self.compute_pairwise_energy())
        if track_magnetization:
            magnetization_evolution = magnetization_evolution.write(
                0, self.compute_magnetization())

        def body(i, spin_evolution, energy_evolution, magnetization_evolution):
            _ = self.metropolis_step(beta, num_disturb, theta_max)

            if i % measurement_granularity == 0:
                j = int(i/measurement_granularity)
                if track_spins:
                    spin_evolution = spin_evolution.write(
                        j + 1, self.spin_state)
                if track_energy:
                    energy_evolution = energy_evolution.write(
                        j + 1, self.compute_pairwise_energy())
                if track_magnetization:
                    magnetization_evolution = magnetization_evolution.write(
                        j + 1, self.compute_magnetization())

            return i + 1, spin_evolution, energy_evolution, magnetization_evolution

        i = tf.constant(0)
        _, spin_evolution, energy_evolution, magnetization_evolution = tf.while_loop(
            lambda i, *_: i < sweep_length,
            body,
            loop_vars=[i, spin_evolution,
                       energy_evolution, magnetization_evolution],
        )

        result = {}
        if track_spins:
            result["spin_evolution"] = spin_evolution.stack()
        if track_energy:
            result["energy_evolution"] = energy_evolution.stack()
        if track_magnetization:
            result["magnetization_evolution"] = magnetization_evolution.stack()

        return result

    # TODO: Refeed the previous spin_state

    @tf.function
    def multi_temperature_sweep(
        self,
        betas: Union[tf.Tensor, np.ndarray],
        num_disturb: Optional[tf.Tensor] = 1,
        theta_max: Optional[tf.Tensor] = None,
        sweep_length: int = 100,
        track_spins: bool = False,
        track_energy: bool = False,
        track_magnetization: bool = True,
    ):
        """
        Perform independent Metropolis sweeps for multiple inverse temperatures (betas).
        Returns:
            dict with keys 'spin_evolution', 'energy_evolution', 'magnetization_evolution',
            each stacked along the first dimension corresponding to beta index.
        """
        betas = tf.convert_to_tensor(betas, dtype=tf.float32)
        n_temps = tf.shape(betas)[0]

        def make_array(track, size):
            return tf.TensorArray(dtype=tf.float32, size=size) if track else tf.TensorArray(dtype=tf.float32, size=0)

        spin_array = make_array(track_spins, n_temps)
        energy_array = make_array(track_energy, n_temps)
        mag_array = make_array(track_magnetization, n_temps)

        def body(i, spin_array, energy_array, mag_array):
            beta = betas[i]

            # Save original spin state
            original_spin_state = tf.identity(self.spin_state)

            # Run a full sweep
            results = self.metropolis_sweep(
                beta=beta,
                num_disturb=num_disturb,
                theta_max=theta_max,
                sweep_length=sweep_length,
                track_spins=track_spins,
                track_energy=track_energy,
                track_magnetization=track_magnetization,
            )

            if track_spins:
                spin_array = spin_array.write(i, results["spin_evolution"])
            if track_energy:
                energy_array = energy_array.write(
                    i, results["energy_evolution"])
            if track_magnetization:
                mag_array = mag_array.write(
                    i, results["magnetization_evolution"])

            # Restore spin state before moving to next beta
            self.spin_state.assign(original_spin_state)

            return i + 1, spin_array, energy_array, mag_array

        i0 = tf.constant(0)
        _, spin_array, energy_array, mag_array = tf.while_loop(
            lambda i, *_: i < n_temps,
            body,
            loop_vars=[i0, spin_array, energy_array, mag_array],
        )

        result = {}
        if track_spins:
            result["spin_evolution"] = spin_array.stack()
        if track_energy:
            result["energy_evolution"] = energy_array.stack()
        if track_magnetization:
            result["magnetization_evolution"] = mag_array.stack()

        return result
