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
