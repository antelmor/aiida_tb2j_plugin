import numpy as np

def get_gamma_matrix(Jiso, M_array, length, indices):

    triangular_values = np.sum(Jiso, axis=1)
    triangular_values *= M_array
    matrix = np.zeros((lenght, lenght))
    matrix[indices] = triangular_values

    return matrix
