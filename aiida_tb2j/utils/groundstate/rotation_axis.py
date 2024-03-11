import numpy as np
from scipy.optimize import basinhopping
from scipy.spatial.transform import Rotation

from ...data.exchange import Hermitize, get_rotation_arrays
from .orientation import angles2cart

def rotate_tensor(vectors, tensor, phi, n):

    np.einsum('ij,k->ijk', phi, n, out=rv)
    R = Rotation.from_rotvec(rv.reshape(-1, 3)).as_matrix().reshape(tensor.shape[:2] + (3, 3))
    np.einsum('nmij,nmjk->nmik', tensor, R, out=rtensor)

def get_Jq(vectors, kvector, idx_pairs):

    exp_summand = np.exp( 2j*np.pi*vectors @ kvector )
    Jexp = exp_summand.reshape( rtensor.shape[:2] + (1, 1) ) * rtensor
    Jq = np.sum(Jexp, axis=1)
    Jq[idx_pairs, :, :] /= 2

    return Jq

def Hermitize(array):

    n = int( (2*array.shape[0])**0.5 )
    result = np.zeros( (n, n) + array.shape[1:], dtype=np.complex128 )
    u_indices = np.triu_indices(n)
    l_indices = np.tril_indices(n)

    result[( *u_indices, )] = array
    lower_entries = np.transpose(result, axes=(1, 0, 3, 2))
    result[( *l_indices, )] = np.conjugate(lower_entries[( *l_indices, )])

    return result

def H_matrix(vectors, kvector, C, U, idx_pairs):

    Jq = -Hermitize( get_Jq(vectors, kvector, idx_pairs) )

    B = np.einsum('ix,ijxy,jy->ij', U, Jq, U)
    A1 = np.einsum('ix,ijxy,jy->ij', U, Jq, U.conjugate())
    A2 = np.einsum('ix,ijxy,jy->ij', U.conjugate(), Jq, U)

    return np.block([
        [A1 - C, B],
        [B.T.conjugate(), A2 - C]
    ])

def optimize_rotation_axis(
        exchange: np.array, 
        Q: np.array = np.zeros(3), 
        kvector: np.array = np.zeros(3), 
        n0: np.array = np.zeros(2), 
        method: str = 'L-BFGS-B', 
        maxiter: int = 180, 
        niter: int = 20, 
        with_Jani: bool = True, 
        with_DMI: bool = True, 
        verbosity: bool = False
    ):

    idx = sorted( set([pair[0] for pair in exchange.pairs]) )
    if exchange.non_collinear:
        magmoms = exchange.magmoms()[idx]
    else:
        magmoms = np.zeros((len(idx), 3))
        magmoms[:, 2] = exchange.magmoms()[idx]
    magmoms /= np.linalg.norm(magmoms, axis=-1).reshape(-1, 1)

    pairs = np.array(exchange.pairs)
    idx_pairs = np.where(pairs[:, 0] == pairs[:, 1])

    vectors = exchange.get_vectors()
    tensor = 1000*exchange.get_exchange_tensor(with_Jani=with_Jani, with_DMI=with_DMI)

    global rtensor, rv
    rtensor = np.empty(tensor.shape)
    rv = np.empty(tensor.shape[:-1])

    phi = 2*np.pi* vectors.round(3).astype(int) @ Q
    U, V = get_rotation_arrays(magmoms)
    J0 = get_Jq(vectors, np.zeros(3), idx_pairs)
    J0 = -Hermitize( J0 )
    C = np.diag( np.einsum('ix,ijxy,jy->i', V, 2*J0, V) )

    def magnon_energy(rot_axis):
        n = angles2cart(rot_axis.reshape(-1, 2)).reshape(3)
        rotate_tensor(vectors, tensor, phi, n)
        H = H_matrix(vectors, kvector, C, U, idx_pairs)
        w = np.linalg.eigvalsh(H)
        return np.min(w)

    bounds = [(0.0, np.pi), (0.0, 2*np.pi)] if exchange.pbc[-1] else [(np.pi/2, np.pi/2), (0.0, 2*np.pi)]
    minimizer_options = {
        'options': {'maxiter': maxiter},
        'method': method,
        'bounds': bounds,
    }

    def callback_fun(x, f, accepted):
        if verbosity:
            print(f'Function value: {f}, Angles: {x}, Accepted: {accepted}')

    optimize_result = basinhopping(magnon_energy, n0, callback=callback_fun, niter=niter, minimizer_kwargs=minimizer_options)
    angles = optimize_result.x

    n = angles2cart(angles.reshape(-1, 2)).reshape(3)
    if verbosity:
        print(optimize_result)

    return n
