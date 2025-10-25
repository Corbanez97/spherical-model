# Spin System Simulation

A flexible and extensible framework for simulating classical spin systems on arbitrary lattices using the Metropolis-Hastings Monte Carlo algorithm. The code supports a variety of models and interaction structures, enabling the study of statistical mechanics phenomena in both standard and exotic settings.

---

## Hamiltonians and Supported Models

### General Hamiltonian

The energy of a spin configuration is defined by a general quadratic form:
$$
\mathcal{H} = \sum_{pq} J_{pq} \sigma_p \sigma_q
$$
where $\sigma_p$ are spin variables and $J_{pq}$ is the user-supplied interaction matrix. This formulation allows for arbitrary lattice geometries and interaction patterns.

### Available Models

- **Ising Model**  
  Spins $\sigma_i = \pm 1$ on a lattice, with Hamiltonian:
  $$
  \mathcal{H}_{\text{Ising}} = -\sum_{ij} J_{ij} \sigma_i \sigma_j - \sum_i h_i \sigma_i
  $$
  Supports arbitrary interaction matrices and external fields.

- **Spherical Model**  
  Continuous spins constrained by $\sum_i \sigma_i^2 = N$, with Hamiltonian:
  $$
  \mathcal{H}_{\text{Spherical}} = -\sum_{ij} J_{ij} \sigma_i \sigma_j - \sum_i h_i \sigma_i
  $$
  The spherical constraint can be enforced optionally.

- **$\mathbb{Z}_2$ Gauge Model**  
  Defined on 2D lattices with link variables and plaquette interactions. The Hamiltonian is constructed from products of link variables around plaquettes, supporting arbitrary coupling matrices.
  $$
  \mathcal{H}_{\text{Z2 Gauge}} = -\sum_{p} \prod_{\ell \in p} \sigma_\ell
  $$

---

## Monte Carlo Method

The simulation uses the **Metropolis-Hastings algorithm** to sample spin configurations according to the Boltzmann distribution at inverse temperature $\beta$. The update scheme is model-dependent:

- **Ising Model:** Random single-spin flips.
- **Spherical Model:** Random pairwise spin rotations, optionally enforcing the spherical constraint.
- **$\mathbb{Z}_2$ Gauge Model:** Random flips of gauge links.

Energy changes are computed efficiently using local field updates and explicit matrix operations.

---

## Features

- Arbitrary lattice dimension and size.
- Multiple replicas for parallel simulation.
- User-defined interaction matrices and external fields.
- Observable tracking: energy, magnetization, overlap matrix, plaquette values, Wilson loops (for gauge models).
- Multi-temperature sweeps for studying phase transitions.
- TensorFlow-based for efficient computation and automatic differentiation.

---

## Usage

1. **Define the interaction matrix** $J_{pq}$ and (optionally) the external field.
2. **Initialize** a `SpinSystem` with desired parameters (lattice size, model type, etc.).
3. **Run simulations** using `metropolis_sweep` or `multi_temperature_sweep`.
4. **Analyze observables** such as energy, magnetization, and overlaps.

---

## References

- Sandvik, A. W. (2010). Computational Studies of Quantum Spin Systems.
- Altieri, A. (2024). Introduction to the Theory of Spin Glasses.
- Bienzobaz, P. (2012). Quantização Canônica e Integração Funcional.

---

For more details and examples, see the [project repository](https://github.com/Corbanez97/spherical-model).