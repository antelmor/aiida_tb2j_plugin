import numpy as np
from itertools import combinations

def get_symmetric_sites(
        q_vector: np.array, 
        coefficients: np.array
    ):

    symmetry_values = (coefficients @ q_vector).round(5)
    indices = np.where(symmetry_values % 1 == 0.0)[0]

    for i, j, k in combinations(indices, 3):
        if np.linalg.det([coefficients[i], coefficients[j], coefficients[k]]).round(5) != 0.0:
            matrix = np.array([coefficients[i], coefficients[j], coefficients[k]])
            break

    try:
        return matrix
    except UnboundLocalError:
        return np.inf*np.ones((3, 3))

def get_transformation_matrix(
        kpoints: np.array, 
        coefficients: np.array
    ):

    possible_sites = np.array([get_symmetric_sites(vector, coefficients) for vector in kpoints])
    determinants = np.abs( np.linalg.det(possible_sites) )
    min_index = np.where( determinants == np.min(determinants) )[0][0]
    transformation_matrix = possible_sites[min_index]
    min_kpoint = kpoints[min_index]

    if np.linalg.det(transformation_matrix) < 0.0:
        transformation_matrix[(0, 1), :] = transformation_matrix[(1, 0), :]

    return transformation_matrix, min_kpoint
