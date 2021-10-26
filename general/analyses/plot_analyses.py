import itertools as it
from numba import jit
import numpy as np


def orthogonality_of_vectors(matrix_of_vectors):
    """Calculate a matrix with the orthogonality of all combinations of the
    vectors in the incomming matrix. Pair of vectors are orthogonal if the value
    is close/equal to zero.

    Parameters
    ----------
    matrix_of_vectors : numpy.ndarray
        A matrix with vectors as rows

    Returns
    -------
    numpy.ndarray
        The resulting orthogonality matrix
    """
    n_vectors = matrix_of_vectors.shape[0]

    # print("matrix_of_vectors", matrix_of_vectors.shape)
    # input()
    # Calculate orthogonality of all combinations
    orthonormality = [x.dot(y) for x, y in it.combinations(matrix_of_vectors, 2)]

    orthogonality_matrix = np.zeros((n_vectors, n_vectors))
    orthogonality_matrix[np.triu_indices(n_vectors, k=1)] = np.abs(orthonormality)

    return orthogonality_matrix
