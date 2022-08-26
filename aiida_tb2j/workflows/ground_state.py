from aiida import orm
from aiida.engine import WorkChain, while_, ToContext, calcfunction
from aiida.common import AttributeDict
from . import TB2JSiestaWorkChain  
from ..data import ExchangeData
from ..utils import get_transformation_matrix
import numpy as np

def validate_scale_processes_options(value, _):

    required_keys = ['max_mpiprocs_per_machine', 'tot_num_mpiprocs']
    present_keys = []
    if value:
        scale_options = value.get_dict()
        for k, v in scale_options.items():
            if k in required_keys:
                if not isinstance(v, int):
                    return f"The '{k}' option must be an integer."
                elif v <= 0:
                    return f"The '{k}' option must be a positive integer."
                else:
                    present_keys.append(k)
            else:
                return f"Unrecognized option '{k}'."
        for key in required_keys:
            if key not in present_keys:
                return f"The {key} option must be present in the 'scale_processes_options' input variable."


def generate_coefficients(size):

    coefficients = np.stack(
        np.meshgrid(*[np.arange(number+1) for number in size]), 
        axis=-1
    ).reshape(-1, 3)[1:]

    return coefficients[coefficients[:, 2].argsort(kind='stable')]

@calcfunction
def is_gamma(
        point: orm.List 
    ) -> orm.Bool:

    point_is_gamma = point == [0.0, 0.0, 0.0]

    return orm.Bool( point_is_gamma )

class GroundStateWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(TB2JSiestaWorkChain, exclude=('metadata',))

        spec.input(
            'tolerance_options',
            valid_type=orm.Dict,
            required=False,
            help='Energy and supercell tolerance values regarding the magnetic ground state search. For more informtaion, read the documentation.'
        )
        spec.input(
            'scale_processes_options',
            valid_type=orm.Dict,
            required=False,
            help='If present, the number of processes will be scaled based on the increased number of atoms of the new generated structures.'
        )

        spec.outline(
            cls.initialize,
            while_(cls.minimum_not_gamma)(
                cls.calculate_Jiso,
                cls.analyze,
            ),
            cls.return_results
        )

        spec.output('exchange', valid_type=ExchangeData, required=True)
        spec.output('converged', valid_type=orm.Bool, required=True)

        spec.exit_code(300, 'ERROR_EXCHANGE_WC', message="The TB2JSiestaWorkChain failed.")

    def initialize(self):

        structure = self.inputs.structure

        try:
            max_iterations = int( self.inputs.tolerance_options['max_iterations'] )
            if max_iterations < 1:
                raise ValueError("The 'max_iterations' value must be a positive integer.")
        except ValueError:
            raise ValueError("The 'max_iterations' value must be an integer.")
        except (AttributeError, KeyError):
            max_iterations = 5

        try:
            energy_tolerance = float( self.inputs.tolerance_options['energy'] )
            if energy_tolerance < 0:
                raise ValueError("The 'energy' value of the tolerance options must be nonnegative.")
        except ValueError:
            raise ValueError("The 'energy' value of the tolerance options must be a number.")
        except (AttributeError, KeyError):
            energy_tolerance = 1e-3

        try:
            max_supercell = list( self.inputs.tolerance_options['max_supercell'] )
            if len(max_supercell) != 3 or not all(isinstance(number, int) for number in max_supercell):
                raise ValueError("The 'max_supercell' value should be a list with 3 integers.")
            elif any(number <= 0 for number in max_supercell):
                raise ValueError("The supercell dimensions should be positive integers.")
        except TypeError:
            raise ValueError("The 'max_supercell' value should be a list with 3 integers.")
        except (AttributeError, KeyError):
            max_supercell = [7 if periodic_condition else 1 for periodic_condition in structure.pbc]
        if any(number > 9 for number in max_supercell):
            self.report(
                "WARNING: The maximum allowed supercell size is too big."
            )
        
        self.ctx.structure = structure
        self.ctx.base_rcell = 2*np.pi* np.linalg.inv(structure.cell).T
        self.ctx.parameters = self.inputs.parameters
        self.ctx.kpoints = self.inputs.kpoints
        self.ctx.options = self.inputs.options

        self.ctx.coefficients = generate_coefficients(max_supercell) 
        self.ctx.energy_tolerance = energy_tolerance
        self.ctx.max_iterations = max_iterations
        self.ctx.iterations = 0
        self.ctx.trans_matrix = np.zeros((3, 3))
        self.ctx.limits = np.array(max_supercell)
        self.ctx.min_kpoint = np.ones(3)

    def minimum_not_gamma(self):

        if (np.linalg.norm(self.ctx.trans_matrix, axis=1) > self.ctx.limits).any():
            self.report(
                "The resulting magnetic configuration exceeds the maximum allowed supercell size. Stopping the WorkChain..."
            )
            return False
        elif self.ctx.iterations >= self.ctx.max_iterations:
            self.report(
                "The magnetic groundstate has not been found in the maximum allowed iterations."
            )
            return False

        return not ( self.ctx.min_kpoint == np.zeros(3) ).all()

    def calculate_Jiso(self):

        inputs = AttributeDict( self.exposed_inputs(TB2JSiestaWorkChain) )
        inputs['structure'] = self.ctx.structure
        inputs['parameters'] = self.ctx.parameters
        inputs['kpoints'] = self.ctx.kpoints
        inputs['options'] = self.ctx.options
        running = self.submit(TB2JSiestaWorkChain, **inputs)
        self.report("Launched TB2JSiestaWorkChain<{}> to calculate the isotropic exchange constants".format(running.pk))

        return ToContext(workchain_exchange=running)

    def _get_new_structure(self):

        from ase.build.supercells import make_supercell

        T = self.ctx.trans_matrix
        ase_atoms = self.inputs.structure.get_ase()
        ase_supercell = make_supercell(ase_atoms, T)

        new_structure = orm.StructureData()
        new_structure.set_ase(ase_supercell)

        return new_structure

    def _get_new_parameters(self):

        from aiida_siesta.utils.tkdict import FDFDict

        parameters = self.inputs.parameters.get_dict()
        new_parameters = FDFDict(parameters).get_dict()
        try:
            del new_parameters['spinpolarized']
        except KeyError:
            pass

        exchange_data = self.ctx.workchain_exchange.outputs.exchange
        positions = np.array([site.position for site in self.ctx.structure.sites])
        ratio = int( len(positions)/len(self.inputs.structure.sites) )
        q_vector = self.ctx.min_kpoint @ self.ctx.base_rcell
        
        magmoms = self.ctx.magmoms*ratio 
        magmoms = np.array(
            [np.linalg.norm(vector) for vector in magmoms]
        )
        phi = positions @ q_vector
        m_x = (magmoms * np.cos(phi)).round(4)
        m_y = (magmoms * np.sin(phi)).round(4)
        deg_phi = (180/np.pi* phi % 360).round(4)

        if m_x.any() and m_y.any():
            if parameters['spin'] != 'spin-orbit':
                new_parameters['spin'] = 'non-collinear'
            init_spin = ''.join(
                [f'\n {i+1} {magmoms[i]} 90.0 {deg_phi[i]}' for i in range( len(phi) )]
            )
        else:
            new_parameters['spin'] = 'polarized'
            if m_x.any():
                init_spin = ''.join(
                    [f'\n {i+1} {m_x[i]}' for i in range( len(m_x) )]
                )
            else:
                init_spin = ''.join(
                    [f'\n {i+1} {m_y[i]}' for i in range( len(m_y) )]
                )
        new_parameters['%block dminitspin'] = init_spin + '\n%endblock dm-init-spin'

        return orm.Dict( dict=new_parameters )

    def _get_new_kpoints(self):

        ratio = np.linalg.norm(self.ctx.structure.cell, axis=-1) / np.linalg.norm(self.inputs.structure.cell, axis=-1)
        new_kmesh = np.ceil( np.array(self.inputs.kpoints.get_kpoints_mesh()[0]) / ratio )
        new_kpoints = orm.KpointsData()
        new_kpoints.set_kpoints_mesh(new_kmesh)

        return new_kpoints

    def _get_new_options(self):

        ratio = int( len(self.ctx.structure.sites)/len(self.inputs.structure.sites) )
        new_options = self.inputs.options.get_dict()
        new_num_mpiprocs = ratio * self.inputs.scale_processes_options['tot_num_mpiprocs']
        max_mpiprocs = self.inputs.scale_processes_options['max_mpiprocs_per_machine']
        
        divisor = 1
        while True:
            tot_num_mpiprocs = new_num_mpiprocs + (divisor - new_num_mpiprocs % divisor) % divisor
            num_mpiprocs_per_machine = int(tot_num_mpiprocs / divisor)
            if num_mpiprocs_per_machine <= max_mpiprocs:
                break
            divisor += 1

        new_options['resources'] = {
            'tot_num_mpiprocs': tot_num_mpiprocs,
            'num_mpiprocs_per_machine': num_mpiprocs_per_machine
        }
        try:
            new_options['max_memory_kb'] *= ratio
        except KeyError:
            pass

        return orm.Dict(dict=new_options)



    def analyze(self):

        if not self.ctx.workchain_exchange.is_finished_ok:
            return self.exit_codes.ERROR_EXCHANGE_WC

        exchange_data = self.ctx.workchain_exchange.outputs.exchange
        rcell = exchange_data.reciprocal_cell()
        min_kpoints = exchange_data.find_minimum_kpoints(tolerance=self.ctx.energy_tolerance, pbc=self.inputs.structure.pbc)
        
        if self.ctx.iterations == 0:
            self.ctx.magmoms = exchange_data.magmoms()

        kpoints = np.linalg.solve(self.ctx.base_rcell.T, (min_kpoints @ rcell).T).T
        T, q = get_transformation_matrix(kpoints, self.ctx.coefficients)
        self.ctx.trans_matrix = T
        self.ctx.min_kpoint = q
        self.report(
            f"Iteration {self.ctx.iterations}, kpoint with minimum energy: {q.round(4)}"
        )

        new_structure = self._get_new_structure()
        self.ctx.structure = new_structure
        new_parameters = self._get_new_parameters()
        self.ctx.parameters = new_parameters
        new_kpoints = self._get_new_kpoints()
        self.ctx.kpoints = new_kpoints
        if 'scale_processes_options' in self.inputs:
            new_options = self._get_new_options()
            self.ctx.options = new_options
        self.ctx.iterations += 1

    def _is_converged(self):

        min_kpoint = orm.List( list=self.ctx.min_kpoint.tolist() )

        return is_gamma(min_kpoint)

    def return_results(self):

        from aiida.engine import ExitCode

        final_exchange = self.ctx.workchain_exchange.outputs.exchange
        converged = self._is_converged()

        self.out('exchange', final_exchange)
        self.out('converged', converged)

        self.report('Magnetic GroundState workchain completed.')
        return ExitCode(0) 
