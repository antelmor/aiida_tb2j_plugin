import numpy as np
from pickle import load as pickle_load
from aiida.orm import ArrayData, StructureData
from aiida.orm.nodes.data.structure import Site

uz = np.array([[0.0, 0.0, 1.0]])
I = np.eye(3)

def get_rotation_arrays(magmoms):

    dim = magmoms.shape[0]
    v = magmoms
    n = v[:, [1, 0, 2]]
    n[:, 0] *= -1
    n[:, -1] *= 0
    n /= np.linalg.norm(n, axis=-1).reshape(dim, 1)
    z = np.repeat(uz, dim, axis=0)
    A = np.stack([z, np.cross(n, z), n], axis=1)
    B = np.stack([v, np.cross(n, v), n], axis=1)
    R = np.einsum('nki,nkj->nij', A, B)

    Rnan = np.isnan(R)
    if Rnan.any():
        nanidx = np.where(Rnan)[0]
        R[nanidx] = I
        R[nanidx, 2] = v[nanidx]

    U = R[:, 0] + 1j*R[:, 1]
    V = R[:, 2]

    return U, V

def Hermitize(array):

    n = int( (2*array.shape[1])**0.5 )
    result = np.zeros( array.shape[0:1] + (n, n) + array.shape[2:], dtype=np.complex128 )
    u_indices = np.triu_indices(n)
    l_indices = np.tril_indices(n)

    result[( slice(None), *u_indices )] = array
    lower_entries = np.transpose(result, axes=(0, 2, 1, 4, 3))
    result[( slice(None), *l_indices )] = np.conjugate(lower_entries[( slice(None), *l_indices )])

    return result

def branched_keys(tb2j_keys, npairs):

    from numpy.linalg import norm

    msites = int( (2*npairs)**0.5 )
    branch_size = int( len(tb2j_keys)/msites**2 )
    new_keys = sorted(tb2j_keys, key=lambda x : -x[1] + x[2])[(npairs-msites)*branch_size:]
    new_keys.sort(key=lambda x : x[1:])
    bkeys = [new_keys[i:i+branch_size] for i in range(0, len(new_keys), branch_size)]

    return [sorted(branch, key=lambda x : norm(x[0])) for branch in bkeys]

def correct_content(content, quadratic=False):

    from numpy import zeros

    n = max(content['index_spin']) + 1

    if content['colinear']:
        content['exchange_Jdict'].update({((0, 0, 0), i, i): 0.0 for i in range(n)})
    else:
        shape = {
            'exchange_Jdict': (),
            'Jani_dict': (3, 3),
            'dmi_ddict': (3)
        }
        if quadratic:
            shape['biquadratic_Jdict'] = (2,)
        for data_type in shape:
            content[data_type].update({((0, 0, 0), i, i): zeros(shape[data_type]) for i in range(n)})

class ExchangeData(ArrayData):

    @property
    def cell(self):

        return np.array( self.get_attribute('cell') )

    @cell.setter
    def cell(self, value):

        return self._set_cell(value)

    def _set_cell(self, value):

        from aiida.common.exceptions import ModificationNotAllowed
        from aiida.orm.nodes.data.structure import _get_valid_cell

        if self.is_stored:
            raise ModificationNotAllowed("ExchangeData cannot be modified because it has already been stored.")

        the_cell = _get_valid_cell(value)
        self.set_attribute('cell', the_cell)

    def reciprocal_cell(self):

        return 2*np.pi* np.linalg.inv( self.get_attribute('cell') ).T

    @property
    def pbc(self):

        return self.get_attribute('pbc', default=(True, True, True))

    @pbc.setter
    def pbc(self, value):

        self._set_pbc(value)

    def _set_pbc(self, value):

        from aiida.common.exceptions import ModificationNotAllowed
        from aiida.orm.nodes.data.structure import get_valid_pbc

        if self.is_stored:
            raise ModificationNotAllowed("ExchangeData cannot be modified because it has already been stored.")
        the_pbc = tuple(get_valid_pbc(value))

        self.set_attribute('pbc', the_pbc)

    @property
    def sites(self):

        raw_sites = self.get_attribute('sites')

        return [MagSite(raw=i) for i in raw_sites]

    def _set_sites(self, sites, magmoms):

        from aiida.common.exceptions import ModificationNotAllowed

        if self.is_stored:
            raise ModificationNotAllowed('ExchangeData cannot be modified because it has already been stored.')

        try:
            self.cell
        except AttributeError:
            raise AttributeError('The cell must be set first, then the sites.')
        try:
            if not magmoms:
                magmoms = [None for i in range(len(sites))]
        except ValueError:
            pass
        if len(sites) != len(magmoms):
            raise ValueError('The number of magnetic moments has to coincide with the number of sites.')

        self.set_attribute('sites', [MagSite(site=site, magmom=magmom).get_raw() for site, magmom in zip(sites, magmoms)])

    def set_structure_info(self, structure, magmoms=None):

        if not isinstance(structure, StructureData):
            raise TypeError(f"set_structure_info() argument must be of type 'StructureData', not {type(structure)}")

        self.cell = structure.cell
        self._set_sites(sites=structure.sites, magmoms=magmoms)

    def magmoms(self):

        return np.array([site.magmom for site in self.sites])

    @property
    def magnetic_elements(self):

        return self.get_attribute('magnetic_elements')

    @magnetic_elements.setter
    def magnetic_elements(self, symbols):

        symbols_tuple = tuple(set(symbols))

        return self._set_magnetic_elements(symbols_tuple)

    def _set_magnetic_elements(self, symbols_tuple):

        from aiida.common.exceptions import ModificationNotAllowed
        from aiida.orm.nodes.data.structure import validate_symbols_tuple

        if 'sites' not in self.attributes:
            raise AttributeError("sites must be set first, then magnetic_elements")
        if self.is_stored:
            raise ModificationNotAllowed("ExchangeData cannot be modified because it has already been stored.")

        validate_symbols_tuple(symbols_tuple)

        self.set_attribute('magnetic_elements', tuple(symbols_tuple))
        self._set_index_pairs()

    @property
    def pairs(self):

        return self.get_attribute('pairs')

    def _set_index_pairs(self):

        import itertools

        magnetic_elements = self.magnetic_elements
        sites = self.sites
        indeces = [sites.index(atom) for atom in sites if any(atom.kind_name == element for element in magnetic_elements)]
        index_pairs = list( itertools.combinations_with_replacement(indeces, 2) )

        self.set_attribute('pairs', index_pairs)

    @property
    def non_collinear(self):

        return self.get_attribute('non_collinear')

    @non_collinear.setter
    def non_collinear(self, value):

        self.set_attribute('non_collinear', bool(value))

    def _equivalent_magsites(self, roundup=2):

        magmoms = self.magmoms().round(roundup)


    @property
    def units(self):

        return self.get_attribute('units', None)

    @units.setter
    def units(self, value):

        self.set_attribute('units', str(value))

    def set_vectors(self, vectors, cartesian=False):

        try:
            pairs = self.get_attribute('pairs')
            pairs_number = len(pairs)
        except KeyError:
            raise ValueError("magnetic_elements must be set first, then vectors")
        try:
            the_vectors = np.array(vectors, dtype=np.float64)
        except ValueError:
            raise ValueError("vectors must be an array of numerical data")
        if len(the_vectors) != pairs_number or the_vectors.shape[1:][1:] != (3,):
            raise ValueError(f"vectors must be an array of dimension 3 with shape (n, m, 3), where n={pairs_number} is the number of exchange pairs")

        if cartesian:
            the_vectors = np.linalg.solve(self.cell.T, np.moveaxis(the_vectors, 1, -1))
            the_vectors = np.moveaxis(the_vectors, -1, 1)

        self.set_array('vectors', the_vectors)

    def get_vectors(self):

        try:
            return self.get_array('vectors')
        except KeyError:
            raise AttributeError("No stored 'vectors' have been found")

    def _validate_exchange_array(self, name, values):

        try:
            vectors = self.get_array('vectors')
        except KeyError:
            raise AttributeError("vectors must be set first, then any exchange array")
        try:
            the_array = np.array(values, dtype=np.float64)
        except ValueError:
            raise ValueError("The exchange array must contain numerical data only.")
        
        if name == 'Jiso':
            if the_array.ndim != 2:
                raise ValueError("The Jiso array must be of dimension 2.")
        elif name == 'Jani':
            if the_array.shape[1:][1:] != (3, 3):
                raise ValueError("The Jani array must have the shape (n, m, 3, 3).")
        elif name == 'DMI':
            if the_array.shape[1:][1:] != (3,):
                raise ValueError("The DMI array must have the shape (n, m, 3).")
        elif name == 'Biquad':
            if the_array.shape[1:][1:] != (2,):
                raise ValueError("The Biquad array must have the shape (n, m, 2).")
        else:
            raise ValueError(f"Unrecognized option '{name}'.")

        if the_array.shape[0:2] != vectors.shape[0:2]:
            raise ValueError(f"The shape of the {name} array must equal the shape of the vectors array on its first two digits.")

        return the_array

    def set_exchange_array(self, name, values):

        the_array = self._validate_exchange_array(name, values)
        self.set_array(name, the_array)

    def get_Jiso(self):

        try:
            return self.get_array('Jiso')
        except KeyError:
            raise AttributeError("No stored 'Jiso' has been found.")

    def get_Jani(self):

        try:
            return self.get_array('Jani')
        except KeyError:
            raise AttributeError("No stored 'Jani' has been found.")

    def get_DMI(self):

        try:
            return self.get_array('DMI')
        except KeyError:
            raise AttributeError("No stored 'DMI' has been found.")

    def get_Biquad(self):

        try:
            return self.get_array('Biquad')
        except KeyError:
            raise AttributeError("No stored 'Biquad' has been found.")

    def get_exchange_tensor(self, with_Jani=True, with_DMI=True):

        try:
            Jiso = self.get_array('Jiso')
        except KeyError:
            raise AttributeError("The array 'Jiso' must be set before calculating the exchange tensor.")

        arraynames = self.get_arraynames()
        diag_indices = ([0, 1, 2], [0, 1, 2])
    
        if with_Jani and 'Jani' in arraynames:
            tensor = self.get_Jani().copy()
        else:
            tensor = np.zeros(Jiso.shape + (3, 3))
        tensor[( slice(None), slice(None), *diag_indices )] += Jiso.reshape(Jiso.shape + (1,))

        if with_DMI and 'DMI' in arraynames:
            pos_indeces = ([1, 2, 0], [2, 0, 1])
            neg_indeces = ([2, 0, 1], [1, 2, 0])
            DMI = self.get_DMI()
            tensor[( slice(None), slice(None), *pos_indeces )] += DMI
            tensor[( slice(None), slice(None), *neg_indeces )] -= DMI

        return tensor

    def _Jq(self, kpoints, with_Jani=False, with_DMI=False):

        vectors = self.get_vectors()
        tensor = self.get_exchange_tensor(with_Jani, with_DMI)
        exp_summand = np.exp( 2j*np.pi*vectors @ kpoints.T ).T
        Jexp = exp_summand.reshape( (kpoints.shape[0], 1, 1) + exp_summand.shape[1:] ) * tensor.T
        Jq = np.sum(Jexp, axis=3)

        return np.transpose(Jq, axes=(0, 3, 2, 1))

    def _H_matrix(self, kpoints, with_Jani=False, with_DMI=False):

        idx = sorted( set([pair[0] for pair in self.pairs]) )
        if self.non_collinear:
            magmoms = self.magmoms()[idx]
        else:
            magmoms = np.zeros((len(idx), 3))
            magmoms[:, 2] = self.magmoms()[idx]
        magmoms /= np.linalg.norm(magmoms, axis=-1).reshape(-1, 1)

        U, V = get_rotation_arrays(magmoms)

        J0 = self._Jq(np.array([[0, 0, 0]]), with_Jani, with_DMI)
        J0 = -Hermitize( J0 )
        Jq = -Hermitize( self._Jq(kpoints, with_Jani, with_DMI) )

        C = np.diag( np.einsum('ix,ijxy,jy->i', V, 2*J0[0], V) )
        B = np.einsum('ix,kijxy,jy->kij', U, Jq, U)
        A1 = np.einsum('ix,kijxy,jy->kij', U, Jq, U.conjugate())
        A2 = np.einsum('ix,kijxy,jy->kij', U.conjugate(), Jq, U)

        return np.block([
            [A1 - C, B],
            [np.transpose(B, axes=(0, 2, 1)).conjugate(), A2 - C]
        ])

    def _magnon_energies(self, kpoints, with_Jani=False, with_DMI=False):

        H = self._H_matrix(kpoints, with_Jani, with_DMI)
        n = int( H.shape[-1] / 2 )
        I = np.eye(n)

        min_eig = 0.0
        try:
            K = np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            try:
                K = np.linalg.cholesky( H + 1e-7*np.eye(2*n) )
            except np.linalg.LinAlgError:
                import warnings
                min_eig = np.min( np.linalg.eigvalsh(H) )
                K = np.linalg.cholesky( H - (min_eig - 1e-7)*np.eye(2*n) )
                warnings.warn(
                "WARNING: The system may be far from the magnetic ground-state. The magnon energies might be unphysical."
                )

        g = np.block([
                [1*I, 0*I],
                [0*I,-1*I]
            ])
        KH = np.transpose(K, axes=(0, 2, 1)).conjugate()

        return np.linalg.eigvalsh( KH @ g @ K )[:, n:] + min_eig

    def get_magnon_bands(
            self,
            kpoints: np.array = np.array([]), 
            path: str = None,
            npoints: int = 100,
            special_points: dict = None,
            tol: float = 2e-4,
            pbc: tuple = None,
            cartesian: bool = False,
            labels: list = None,
            with_Jani: bool = False,
            with_DMI: bool = False
    ):

        from aiida.orm import BandsData
        bands_data = BandsData()

        if pbc == None:
            pbc = self.pbc

        if not kpoints.any():
            from ase.cell import Cell
            bandpath = Cell(self.cell).bandpath(
                path=path, 
                npoints=npoints, 
                special_points=special_points,
                eps=tol,
                pbc=pbc
            )
            kpoints = bandpath.kpts
            spk = bandpath.special_points
            try:
                spk['GAMMA'] = spk.pop('G')
            except KeyError:
                pass
            labels = [(i, symbol) for symbol in spk for i in np.where( (kpoints == spk[symbol]).all(axis=1) )[0]]
        if cartesian:
            bands_data.cell = self.cell
        
        bands_data.set_kpoints(kpoints, cartesian=cartesian, labels=sorted(labels))
        magnon_energies = self._magnon_energies( bands_data.get_kpoints(), with_Jani, with_DMI )
        bands_data.set_bands(magnon_energies, units=self.units)

        return bands_data

    def find_minimum_kpoints(
            self, 
            kpoints: np.array = np.array([]),
            tolerance: float = 0,
            pbc: tuple = None,
            with_Jani: bool = False,
            with_DMI: bool = False,
            size: list = None
    ):

        if pbc == None:
            pbc = self.pbc

        if not kpoints.any():
            from TB2J.kpoints import monkhorst_pack
            if size is None:
                size = np.ones(3, dtype=int)
                size[list(pbc)] = 16
            kpoints = monkhorst_pack(size, gamma_center=True)

        n = int( (2*len(self.pairs))**0.5 )
        H = self._H_matrix( kpoints, with_Jani, with_DMI )
        eigenvals = np.linalg.eigvalsh( H )
        min_eig = np.min(eigenvals)
        
        minimum_indices = np.where(eigenvals - min_eig <= tolerance)[0]
 
        return kpoints[minimum_indices]

    def get_structure(self):

        structure = StructureData()
        structure.cell = self.cell
        structure.pbc = self.pbc
        for atom in self.sites:
            structure.append_atom(position=atom.position, symbols=atom.kind_name)

        return structure

    @classmethod
    def load_tb2j(cls, content=None, pickle_file='TB2J.pickle', pbc=None, isotropic=True, quadratic=False):

        if content is None:
            try:
                with open(pickle_file, 'rb') as File:
                    content = pickle_load(File)
                    correct_content(content)
            except FileNotFoundError:
                raise FileNotFoundError(f"No such file or directory: '{pickle_file}'. Please provide a valid .pickle file.")

        exchange = cls()
        structure = StructureData(ase=content['atoms'])
        
        if content['colinear']:
            exchange.non_collinear = False
            exchange.set_structure_info(structure=structure, magmoms=content['magmoms'])
        else:
            exchange.non_collinear = True
            exchange.set_structure_info(structure=structure, magmoms=content['spinat'])
        exchange.magnetic_elements = [exchange.sites[i].kind_name for i in range(len(exchange.sites)) if content['index_spin'][i] >= 0]
        if pbc is not None:
            exchange.pbc = pbc

        bkeys = branched_keys(content['distance_dict'].keys(), len(exchange.pairs))
        vectors = [ [content['distance_dict'][key][0] for key in branch] for branch in bkeys ]
        exchange.set_vectors(vectors, cartesian=True)
            
        Jiso = [ [content['exchange_Jdict'][key] for key in branch] for branch in bkeys ]
        exchange.set_exchange_array('Jiso', Jiso)
            
        if exchange.non_collinear and not isotropic:
           Jani = [ [content['Jani_dict'][key] for key in branch] for branch in bkeys ]
           exchange.set_exchange_array('Jani', Jani)
           DMI = [ [content['dmi_ddict'][key] for key in branch] for branch in bkeys ]
           exchange.set_exchange_array('DMI', DMI)
           if quadratic:
               Biquad = [ [content['biquadratic_Jdict'][key] for key in branch] for branch in bkeys ]
               exchange.set_exchange_array('Biquad', Biquad)

        return exchange
        

class MagSite(Site):

    def __init__(self, **kwargs):

        self._magmom = None

        try:
            self.magmom = kwargs.pop('magmom')
        except KeyError:
            pass

        super().__init__(**kwargs)

        try:
            self.magmom = kwargs['raw']['magmom']
        except KeyError:
            pass

    @property
    def magmom(self):

        return self._magmom

    @magmom.setter
    def magmom(self, value):

        try:
            magmom = np.array(value, dtype=np.float64)
        except ValueError:
            raise ValueError("The argument 'magmom' only accepts numerical data.")
        if magmom.ndim != 0 and magmom.shape != (3,):
            raise ValueError("The magmom value should be either a single number or an array of 3 numbers.")
        self._magmom = magmom

    def get_raw(self):

        return {
            'position': self.position,
            'kind_name': self.kind_name,
            'magmom': self.magmom.tolist()
        }
