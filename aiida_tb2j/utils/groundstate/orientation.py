import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import basinhopping
from aiida.orm import Dict

from ...data.exchange import Hermitize, get_rotation_arrays

uz = np.array([[0.0, 0.0, 1.0]])
I = np.eye(3)

def angles2cart(array):

    s = np.sin(array[:, 0])
    new_array = np.empty((array.shape[0], 3))
    new_array[:, 0] = np.cos(array[:, 1])*s
    new_array[:, 1] = np.sin(array[:, 1])*s
    new_array[:, 2] = np.cos(array[:, 0])

    return new_array

def cart2spher(array):

    new_array = np.empty(array.shape)
    xy = array[:,0]**2 + array[:,1]**2
    new_array[:,0] = np.sqrt(xy + array[:,2]**2)
    new_array[:,1] = 180/np.pi*np.arctan2(np.sqrt(xy), array[:,2])
    new_array[:,2] = 180/np.pi*np.arctan2(array[:,1], array[:,0])
   
    return new_array

def H0_matrix(magmoms, J0):

    i, _, x, _ = J0.shape
    U, V = get_rotation_arrays(magmoms)
    U = U.reshape(-1, i, x)
    V = V.reshape(-1, i, x)
    Uc = U.conjugate()

    C_a = np.einsum('kix,ijxy,kjy->ki', V, 2*J0, V)
    C = np.einsum('ki,ij->kij', C_a, np.eye(i))
    B = np.einsum('kix,ijxy,kjy->kij', U, J0, U)
    A1 = np.einsum('kix,ijxy,kjy->kij', U, J0, Uc)
    A2 = np.einsum('kix,ijxy,kjy->kij', Uc, J0, U)
    
    return np.block([
        [A1 - C, B],
        [B.swapaxes(1, 2).conjugate(), A2 - C]
    ])

def find_orientation(
        exchange: np.array, 
        x0: np.array = None, 
        method: str = 'L-BFGS-B', 
        maxiter: int = 180, 
        niter: int = 20, 
        threshold: float = 1e-3, 
        magmom_threshold: float = 0.0,
        Q: np.array = None, 
        kpoint: np.array = np.zeros(3), 
        with_DMI: bool = True, 
        with_Jani: bool = True,
        verbosity: bool = False
    ):

    idx = sorted( set([pair[0] for pair in exchange.pairs]) )
    if exchange.non_collinear:
        final_magmoms = exchange.magmoms()
    else:
        final_magmoms = np.zeros((len(exchange.sites), 3))
        final_magmoms[:, 2] = exchange.magmoms()
    magmoms = final_magmoms[idx]

    magnorm = np.linalg.norm(magmoms, axis=-1)
    J0 = 1e+3*exchange._Jq(np.array([kpoint]), with_Jani=with_DMI, with_DMI=with_Jani, Q=Q)
    J0 = -Hermitize( J0 )[0]

    jdx = np.where(magnorm >= magmom_threshold)[0]

    def eval_gamma(angles):
        magmoms[jdx] = angles2cart(angles.reshape(-1, 2))
        H0 = H0_matrix(magmoms, J0)
        w = np.linalg.eigvalsh(H0)
        return np.abs( np.min(w, axis=-1) )
    
    if x0 is None:
        x0 = np.pi/180* cart2spher(magmoms[jdx])[:, (1, 2)].reshape(-1)
    options = {
        'maxiter': maxiter
    }
    minimizer_options = {
        'options': options,
        'method': method,
        'bounds': [(0.0, np.pi), (0.0, 2*np.pi)]*len(jdx)
    }
    def stop_fun(x, f, accepted):
        if verbosity:
            print(f"Function value: {f},  Accepted: {accepted}")
        return abs(f) < threshold

    optimize_result = basinhopping(eval_gamma, x0, callback=stop_fun, niter=niter, minimizer_kwargs=minimizer_options)
    opt_angles = optimize_result.x
    magmoms[jdx] = magnorm[jdx].reshape(-1, 1) * angles2cart(opt_angles.reshape(-1, 2))

    if verbosity:
        print(optimize_result)

    final_magmoms[idx] = magmoms

    return final_magmoms
