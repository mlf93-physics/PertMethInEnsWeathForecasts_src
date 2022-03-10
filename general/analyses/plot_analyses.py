import itertools as it
import numpy as np
import general.utils.util_funcs as g_utils


def orthogonality_of_vectors(matrix_of_vectors):
    """Calculate a matrix with the abs orthogonality of all combinations of the
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

    # Calculate orthogonality of all combinations
    orthonormality = [
        x.dot(y.conj()).real for x, y in it.combinations(matrix_of_vectors, 2)
    ]

    orthogonality_matrix = np.zeros((n_vectors, n_vectors))
    orthogonality_matrix[np.triu_indices(n_vectors, k=1)] = np.abs(orthonormality)

    return orthogonality_matrix


def orthogonality_to_vector(
    reference_vector: np.ndarray, matrix_of_vectors: np.ndarray
) -> np.ndarray:
    """Calculate the orthogonality between a reference vector and one or more
    'trial' vectors.

    Parameters
    ----------
    reference_vector : np.ndarray((n_vectors, dim))
        The reference vector
    matrix_of_vectors : np.ndarray
        The trial vectors

    Returns
    -------
    np.ndarray
        The orthogonality array
    """

    # Make sure vectors are normalized
    reference_vector = g_utils.normalize_array(
        reference_vector.ravel(), norm_value=1, axis=0
    )
    matrix_of_vectors = g_utils.normalize_array(matrix_of_vectors, norm_value=1, axis=1)
    # Calculate orthogonality
    orthogonality = [
        np.vdot(reference_vector, matrix_of_vectors[i, :]).real
        for i in range(matrix_of_vectors.shape[0])
    ]

    return orthogonality
