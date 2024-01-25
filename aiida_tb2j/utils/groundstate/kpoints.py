import numpy as np
from scipy.optimize import basinhopping

from ...data.exchange import Hermitize, get_rotation_arrays

def get_Jq(vectors, tensor, kpoints):

    exp_summand = np.exp( 2j*np.pi*vectors @ kpoints.T ).T
    Jexp = exp_summand.reshape( (kpoints.shape[0], 1, 1) + exp_summand.shape[1:] ) * tensor.T
    Jq = np.sum(Jexp, axis=3)

    return np.transpose(Jq, axes=(0, 3, 2, 1))

def H_matrix(vectors, tensor, kpoints, C, U):

    Jq = -Hermitize( get_Jq(vectors, tensor, kpoints) )

    B = np.einsum('ix,kijxy,jy->kij', U, Jq, U)
    A1 = np.einsum('ix,kijxy,jy->kij', U, Jq, U.conjugate())
    A2 = np.einsum('ix,kijxy,jy->kij', U.conjugate(), Jq, U)

    return np.block([
        [A1 - C, B],
        [np.transpose(B, axes=(0, 2, 1)).conjugate(), A2 - C]
    ])

def find_minimum_kpoints(
        exchange: np.array, 
        x0: np.array=None, 
        method: str='L-BFGS-B', 
        maxiter: int=180, 
        niter: int=20, 
        with_Jani: bool=True, 
        with_DMI: bool=True, 
        magmoms: np.array=None,
        verbosity: bool=False
    ):

    idx = sorted( set([pair[0] for pair in exchange.pairs]) )
    if magmoms is None:
        if exchange.non_collinear:
            magmoms = exchange.magmoms()[idx]
        else:
            magmoms = np.zeros((len(idx), 3))
            magmoms[:, 2] = exchange.magmoms()[idx]
        magmoms /= np.linalg.norm(magmoms, axis=-1).reshape(-1, 1)

    vectors = exchange.get_vectors()
    tensor = exchange.get_exchange_tensor(with_Jani=with_Jani, with_DMI=with_DMI)
    U, V = get_rotation_arrays(magmoms)

    J0 = exchange._Jq(np.zeros((1, 3)), with_Jani, with_DMI)
    J0 = -Hermitize( J0 )
    C = np.diag( np.einsum('ix,ijxy,jy->i', V, 2*J0[0], V) )

    def magnon_energies(kpoints):
        k = kpoints.reshape(-1, 3)
        H = H_matrix(vectors, tensor, k, C, U)
        w = np.linalg.eigvalsh(H)
        return np.min(w)

    if x0 is None:
        x0 = np.zeros(3)
    options = {
        'maxiter': maxiter
    }
    minimizer_options = {
        'options': options,
        'method': method,
        'bounds': [(-0.5, 0.5) if x else (0.0, 0.0) for x in exchange.pbc]
    }

    def info(x, f, accepted):
        if verbosity:
            print("at minimum %.4f accepted %d" % (f, int(accepted)))

    optimize_result = basinhopping(magnon_energies, x0, callback=info, niter=niter, minimizer_kwargs=minimizer_options)
    min_kpoint = optimize_result.x

    if verbosity:
        print(optimize_result)

    return min_kpoint
