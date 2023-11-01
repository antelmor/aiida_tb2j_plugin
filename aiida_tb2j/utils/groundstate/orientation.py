import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import basinhopping
from aiida.orm import Dict
from aiida_siesta.utils.tkdict import FDFDict
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

def find_orientation(exchange, x0=None, method='L-BFGS-B', maxiter=180, niter=20, threshold=1e-2, magmom_threshold=0.1, verbosity=False, kpoint=np.zeros(3)):

    idx = sorted( set([pair[0] for pair in exchange.pairs]) )
    if exchange.non_collinear:
        magmoms = exchange.magmoms()[idx]
    else:
        magmoms = np.zeros((len(idx), 3))
        magmoms[:, 2] = exchange.magmoms()[idx]

    magnorm = np.linalg.norm(magmoms, axis=-1)
    J0 = 1e+6*exchange._Jq(np.array([kpoint]), with_Jani=True, with_DMI=True)
    J0 = -Hermitize( J0 )[0]

    jdx = np.where(magnorm > magmom_threshold)[0]

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
        'bounds': [(-np.pi, np.pi), (0.0, 2*np.pi)]*len(jdx)
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

    return magmoms

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