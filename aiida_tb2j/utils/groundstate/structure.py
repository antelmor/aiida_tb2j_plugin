import numpy as np
from itertools import product
from itertools import combinations, permutations
from ase.build.supercells import make_supercell
from scipy.spatial.transform import Rotation
from aiida.orm import Dict, StructureData
from ...data import ExchangeData
from .orientation import get_new_parameters

def generate_coefficients(size):

    r = [range(s) for s in size]
    coefficients = sorted(product(*r), key=sum)

    return coefficients[1:]

def get_symmetric_sites(
        q_vector: np.array, 
        coefficients: np.array,
        dim: int = 3
    ):

    symmetry_values = (coefficients @ q_vector).round(5)
    indices = np.where(symmetry_values % 1 == 0.0)
    selected_coefficients = coefficients[indices]

    for matrix in combinations(selected_coefficients, dim):
        if np.linalg.det(matrix).round(5) != 0.0:
            result = np.array(matrix)
            break

    try:
        return result
    except UnboundLocalError:
        return np.diag(np.inf*np.ones(3))

def reorder_rows(T):

    I = np.eye(3)
    P = np.array(list(permutations(T)))
    norm = np.linalg.norm(P - I, axis=(1, 2))
    index = np.where(norm == norm.min())[0]

    return P[index][0]

def get_transformation_matrix(
        kpoints: np.array, 
        coefficients: np.array,
        pbc: tuple = (True, True, True)
    ):

    dim = len([b for b in pbc if b])
    possible_sites = np.array(
        [get_symmetric_sites(vector, coefficients, dim) for vector in kpoints.T[list(pbc)].T]
    )
    determinants = np.abs( np.linalg.det(possible_sites) )
    min_index = np.where( determinants == np.min(determinants) )[0][0]
    min_kpoint = kpoints[min_index]

    symmetric_sites = possible_sites[min_index]
    if np.linalg.det(symmetric_sites) < 0.0:
        symmetric_sites[(0, 1), :] = symmetric_sites[(1, 0), :]
    transformation_matrix = np.eye(3).astype(int)
    mask = np.array([[x and y for y in pbc] for x in pbc])
    transformation_matrix[mask] = np.reshape(symmetric_sites, (-1,))
    T = reorder_rows(transformation_matrix)

    return T, min_kpoint

def groundstate_structure(
        exchange: ExchangeData,
        old_structure: StructureData,
        tolerance: float = 0.0,
        maximum_size: list = None,
        coefficients: np.array = None,
        with_DMI: bool = False,
        with_Jani: bool = False
    ):

    pbc = list(old_structure.pbc)
    base_rcell = 2*np.pi* np.linalg.inv(old_structure.cell).T

    if coefficients is None:
        if maximum_size is None:
            maximum_size = np.array([16 if value else 1 for value in pbc])
        coefficients = generate_coefficients(maximum_size[pbc])

    rcell = exchange.reciprocal_cell()
    min_kpoints = exchange.find_minimum_kpoints(
        tolerance=tolerance, pbc=pbc, size=maximum_size, with_DMI=with_DMI, with_Jani=with_Jani
    )
    kpoints = np.linalg.solve(base_rcell.T, (min_kpoints @ rcell).T).T
    T, q = get_transformation_matrix(kpoints, coefficients, pbc=pbc)

    atoms = old_structure.get_ase()
    supercell = make_supercell(atoms, T)
    new_structure = StructureData(ase=supercell)

    return new_structure, q

def vrepeat(array, reps):

    new_array = array.reshape((1,) + array.shape)
    result = np.repeat(new_array, reps, axis=0)

    return result.reshape(-1, array.shape[-1])

def get_rotated_magmoms(
        magmoms: np.array,
        positions: np.array,
        q_vector: np.array,
        rot_axis: np.array = None
    ):

    magnorm = np.linalg.norm(magmoms, axis=-1)
    ref_index = np.where(magnorm == np.max(magnorm))[0][0]
    ratio = int( len(positions)/len(magmoms) )
    new_magmoms = vrepeat(magmoms, ratio)

    if rot_axis is None:
        rot_axis = np.cross( np.array([0.0, 1.0, 0.0]), magmoms[ref_index] )
        if np.linalg.norm(rot_axis) < 1e-2:
            rot_axis = np.cross( np.array([0.0, 0.0, 1.0]), magmoms[ref_index] )
    rot_axis /= np.linalg.norm(rot_axis)
    rot_axis = vrepeat(rot_axis, len(new_magmoms))

    phi = 2*np.pi* positions @ q_vector
    phi = phi.reshape(phi.shape + (1,))
    R = Rotation.from_rotvec(phi*rot_axis).as_matrix()
    rotated_magmoms = np.einsum('nij,nj->ni', R, new_magmoms)

    return rotated_magmoms

def groundstate_data(
        exchange: ExchangeData,
        parameters: Dict,
        magmoms: np.array = None,
        tolerance: float = 0.0,
        maximum_size: list = None,
        coefficients: np.array = None,
        old_structure: StructureData = None,
        with_DMI: bool = False,
        with_Jani: bool = False,
        spiral_mode: bool = False,
        rot_axis: np.array = None
    ):

    if old_structure is None:
        old_structure = exchange.get_structure()
    ref_cell = np.array(old_structure.cell)

    structure, q_vector = groundstate_structure(
        exchange, old_structure, tolerance, maximum_size, coefficients, with_DMI, with_Jani
    )

    if magmoms is None:
        if exchange.non_collinear:
            magmoms = exchange.magmoms().round(2)
        else:
            magmoms = np.zeros((len(exchange.sites), 3))
            magmoms[:, 2] = exchange.magmoms()
    cart_positions = np.array([site.position for site in structure.sites])
    positions = np.linalg.solve(ref_cell.T, cart_positions.T).T
    if not spiral_mode:
        positions = positions.round(2).astype(np.int32)
    new_magmoms = get_rotated_magmoms(magmoms, positions, q_vector, rot_axis=rot_axis)
    new_parameters = get_new_parameters(new_magmoms, parameters)

    return structure, new_parameters, q_vector
