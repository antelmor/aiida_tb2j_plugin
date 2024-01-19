import numpy as np
from itertools import product, permutations
from fractions import Fraction
from math import lcm
from ase.build.supercells import make_supercell
from scipy.spatial.transform import Rotation
from aiida.orm import Dict, StructureData
#from ...data import ExchangeData
from aiida_tb2j.data import ExchangeData
#from .orientation import find_orientation
from aiida_tb2j.utils.groundstate.orientation import find_orientation
#from .kpoints import find_minimum_kpoints
from aiida_tb2j.utils.groundstate.kpoints import find_minimum_kpoints
#from .rotation_axis import optimize_rotation_axis
from aiida_tb2j.utils.groundstate.rotation_axis import optimize_rotation_axis

def generate_coefficients(size):

    r = [range(s) for s in size]
    coefficients = sorted(product(*r), key=sum)

    return coefficients[1:]

def find_coefficients(integers, multiple):

    positives = [x for x in integers[:-1] if x > 0]
    
    if integers[-1] == 0:
        return [0,]*len(integers[:-1]) + [1,]
    elif not positives:
        c = -1
    else:
        for values in generate_coefficients([multiple,]*len(positives)):
            c = np.dot(values, positives)
            if not c % integers[-1]:
                coefs = list(values)
                c /= integers[-1]
                break
    result = [n if not n else coefs.pop(0) for n in integers[:-1]]

    return result + [c,]

def reorder_rows(T):

    I = np.eye(3)
    P = np.array(list(permutations(T)))
    norm = np.linalg.norm(P - I, axis=(1, 2))
    index = np.where(norm == norm.min())[0]

    return P[index][0]

def get_transformation_matrix(
        q_vector: np.array,
        max_size: np.array=8*np.ones(3)
    ):

    rationals = [Fraction.from_float(x).limit_denominator(N).as_integer_ratio() for x, N in zip(q_vector, max_size)]
    l, m = zip(*rationals)
    c = lcm(*m)
    print(l, m, c)
    k = [int(q_vector[i]*c)%c for i in range(3)]
    print(k)

    T = np.array([
            find_coefficients([k[0]], c) + [0,0],
            find_coefficients([c-k[0], k[1]], c) + [0,],
            find_coefficients([c-k[0], c-k[1], k[2]], c)
    ])
    nidx = np.where(T < 0)
    T[nidx] = np.array(m)[nidx[0]]

    T = reorder_rows(T)

    return T

def groundstate_structure(
        exchange: ExchangeData,
        old_structure: StructureData,
        max_size: list = 8*np.ones(3),
        with_DMI: bool = False,
        with_Jani: bool = False
    ):

    base_rcell = 2*np.pi* np.linalg.inv(old_structure.cell).T

    rcell = exchange.reciprocal_cell()
    min_k = find_minimum_kpoints(exchange, with_DMI=with_DMI, with_Jani=with_Jani)

    q = np.linalg.solve(base_rcell.T, (min_k @ rcell).T).T
    T = get_transformation_matrix(q, max_size)

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
        rot_axis = optimize_rotation_axis(exchange, Q=q_vector, kvector=q_vector)
    rot_axis = vrepeat(rot_axis, len(new_magmoms))

    phi = 2*np.pi* positions @ q_vector
    phi = phi.reshape(phi.shape + (1,))
    R = Rotation.from_rotvec(phi*rot_axis).as_matrix()
    rotated_magmoms = np.einsum('nij,nj->ni', R, new_magmoms)

    return rotated_magmoms

def get_new_parameters(magmoms, parameters):

    param_dict = FDFDict(parameters.get_dict()).get_dict()
    try:
        del param_dict['spinpolarized']
    except KeyError:
        pass
    if 'spin' not in param_dict:
        param_dict['spin'] = 'non-collinear'
    elif param_dict['spin'] != 'spin-orbit':
        param_dict['spin'] = 'non-collinear'

    nspins = len(magmoms)
    m = cart2spher(magmoms.round(2)).round(2)
    init_spin = ''.join(
        [f'\n {i+1} {m[i, 0]} {m[i, 1]} {m[i, 2]}' for i in range(nspins)]
    )
    param_dict['%block dminitspin'] = init_spin + '\n%endblock dminitspin'

    return Dict(dict=param_dict)

def groundstate_data(
        exchange: ExchangeData,
        parameters: Dict,
        magmoms: np.array = None,
        optimize_magmoms: bool = False,
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

    if optimize_magmoms:
        magmoms = find_orientation(exchange) 
    elif magmoms is None:
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
