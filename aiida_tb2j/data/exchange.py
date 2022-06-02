import numpy as np
from aiida.orm import ArrayData
from aiida.orm.nodes.data.structure import Site

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
        the_pbc = get_valid_pbc(value)

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

        from aiida.orm import StructureData

        if not isinstance(structure, StructureData):
            raise TypeError(f"set_structure_info() argument must be of type 'StructureData', not {type(structure)}")

        self.cell = structure.cell
        self._set_sites(sites=structure.sites, magmoms=magmoms)

    def magmoms(self):

        return [site.magmom for site in self.sites]

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

    def _Jq(self, kpoints):

        try:
            Jiso = self.get_array('Jiso')
        except KeyError:
            raise AttributeError("the array 'Jiso' must be set before calculating the magnon energies.")

        vectors = self.get_vectors()
        Jcos = np.cos( 2*np.pi*vectors @ kpoints.T ).T * Jiso.T

        return np.sum(Jcos, axis=1)

    def _magnon_energies(self, kpoints):

        magmoms = self.magmoms()
        value_indices, sign_indices = list(zip(*self.pairs))
        M_array = np.sign(np.take(magmoms, sign_indices)) / np.abs(np.take(magmoms, value_indices))

        gamma_indices = np.where((kpoints == np.zeros(3)).all(1))[0]
        Jq = self._Jq(kpoints)

        n = int( (2*len(self.pairs))**0.5 )
        exchange_tensor = np.zeros((len(kpoints), n, n))
        indices = np.triu_indices(n)
        exchange_tensor[( slice(None), *indices )] = -Jq*M_array
        try:
            gamma_matrix = exchange_tensor[gamma_indices[0]]
        except IndexError:
            from ..utils import get_gamma_matrix
            gamma_matrix = get_gamma_matrix(Jiso, M_array, n, indices)
        gamma_matrix = -np.where(gamma_matrix, gamma_matrix, gamma_matrix.T)
        exchange_tensor += np.diag( np.sum(gamma_matrix, axis=0) )

        return 4*np.linalg.eigvalsh(exchange_tensor, UPLO='U')

    def get_magnon_bands(
            self,
            kpoints: np.array = np.array([]), 
            path: str = None,
            npoints: int = 100,
            special_points: dict = None,
            tol: float = 2e-4,
            cartesian: bool = False,
            labels: list = None
    ):

        from aiida.orm import BandsData
        bands_data = BandsData()

        if not kpoints.any():
            from ase.cell import Cell
            bandpath = Cell(self.cell).bandpath(
                path=path, 
                npoints=npoints, 
                special_points=special_points,
                eps=tol,
                pbc=self.pbc
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
        
        bands_data.set_kpoints(kpoints, cartesian=cartesian, labels=labels)
        magnon_energies = self._magnon_energies( bands_data.get_kpoints() )
        bands_data.set_bands(magnon_energies, units=self.units)

        return bands_data

    def find_minimum_kpoints(
            self, 
            kpoints: np.array = np.array([]),
            tolerance: float = 0,
            pbc: tuple = None
    ):

        if pbc == None:
            pbc = self.pbc

        if not kpoints.any():
            from TB2J.kpoints import monkhorst_pack
            dimensions = np.ones(3, dtype=int)
            dimensions[list(pbc)] = 8 
            kpoints = monkhorst_pack(dimensions, gamma_center=True)

        n = int( (2*len(self.pairs))**0.5 )
        Jq = self._Jq(kpoints)
        J_matrix = np.zeros((len(kpoints), n, n))
        J_matrix[( slice(None), *np.triu_indices(n) )] = Jq

        Jeig = np.linalg.eigvalsh(J_matrix, UPLO='U')        
        minimum_indices = np.where(np.max(Jeig) - Jeig <= tolerance)[0]
 
        return kpoints[minimum_indices]

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
