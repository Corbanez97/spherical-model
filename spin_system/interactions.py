import numpy as np


def decaying(D, L, J0=10, alpha=1):
    coords = np.array(np.meshgrid(
        *[np.arange(L)]*D, indexing='ij')).reshape(D, -1).T

    # Pairwise Euclidean distances
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    J_flat = J0 * np.exp(-alpha * distances)
    np.fill_diagonal(J_flat, 0)

    # Vectorized reshape
    tensor_shape = (L,)*D*2
    J_tensor = J_flat.reshape(tensor_shape)

    return J_tensor


def periodic_nn(D, L):
    """
    Vectorized nearest-neighbor coupling tensor with periodic boundaries.
    J[i1,...,iD,j1,...,jD] = 1 if periodic Manhattan distance = 1, else 0
    """
    # Generate all coordinates: shape (N, D)
    coords = np.array(np.meshgrid(
        *[np.arange(L)]*D, indexing='ij')).reshape(D, -1).T
    N = coords.shape[0]

    # Compute pairwise differences with broadcasting
    diff = np.abs(coords[:, None, :] - coords[None, :, :])

    # Apply periodic boundary
    diff = np.minimum(diff, L - diff)

    # Manhattan distance
    manhattan_dist = diff.sum(axis=2)

    # Nearest neighbors mask
    nn_mask = (manhattan_dist == 1)

    # Create empty tensor and set neighbors
    J_tensor = np.zeros((L,)*D*2, dtype=np.float32)

    # Get indices where nn_mask is True
    idx_i, idx_j = np.nonzero(nn_mask)

    # Set values in the tensor
    for i, j in zip(idx_i, idx_j):
        J_tensor[tuple(coords[i]) + tuple(coords[j])] = 1.0

    return J_tensor


def curie_weiss(D, L, J0=1.0):
    N = L**D

    J_flat = (J0 / N) * (np.ones((N, N)) - np.eye(N))

    tensor_shape = (L,) * D * 2
    J_tensor = J_flat.reshape(tensor_shape)

    return J_tensor


def gaussian_interaction(D, L, mean=0.0, std=1.0):
    N = L**D
    J_flat = np.random.normal(mean, std, size=(N, N))

    J_flat = 0.5 * (J_flat + J_flat.T)

    np.fill_diagonal(J_flat, 0)

    tensor_shape = (L,) * D * 2
    J_tensor = J_flat.reshape(tensor_shape)

    return J_tensor


def z2gauge_interaction_tensor(L, periodic="xy", interaction_strength=1.0):
    """
    Build a 3-layer interaction tensor for 2D Z2 gauge theory on a square lattice.

    Parameters
    ----------
    L : int
        Lattice size (LxL square lattice).
    periodic : str
        Periodic boundary conditions: "x", "y", "xy", or None
    interaction_strength : float
        Value of the interaction for nearest neighbors.

    Returns
    -------
    J_tensor : np.ndarray
        Shape (3, L, L, L, L)
        Layer 0: horizontal-horizontal interactions
        Layer 1: vertical-vertical interactions
        Layer 2: horizontal-vertical interactions (plaquette)
    """
    # Initialize empty tensor
    J_tensor = np.zeros((3, L, L, L, L), dtype=np.float32)

    # Horizontal-horizontal (row neighbors)
    for i in range(L):
        for j in range(L):
            j_next = (j + 1) % L if 'y' in periodic else j + 1
            if j_next < L:
                J_tensor[0, i, j, i, j_next] = interaction_strength

    # Vertical-vertical (column neighbors)
    for i in range(L):
        for j in range(L):
            i_next = (i + 1) % L if 'x' in periodic else i + 1
            if i_next < L:
                J_tensor[1, i, j, i_next, j] = interaction_strength

    # Horizontal-vertical (plaquette interactions)
    for i in range(L):
        for j in range(L):
            i_next = (i + 1) % L if 'x' in periodic else i + 1
            j_next = (j + 1) % L if 'y' in periodic else j + 1
            if i_next < L and j_next < L:
                # Bottom-left horizontal-vertical interaction
                J_tensor[2, i, j, i, j] = interaction_strength
                # Other links forming the plaquette
                # J_tensor[2, i, j, i_next, j] = interaction_strength
                # J_tensor[2, i, j, i, j_next] = interaction_strength
                # J_tensor[2, i, j, i_next, j_next] = interaction_strength

    return J_tensor
