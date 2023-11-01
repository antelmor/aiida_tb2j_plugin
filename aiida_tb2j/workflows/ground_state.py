from aiida import orm
from aiida.engine import WorkChain, while_, ToContext, calcfunction
from aiida.common import AttributeDict
from . import TB2JSiestaWorkChain, MagneticOrientationWorkChain  
from ..data import ExchangeData
from ..utils import generate_coefficients, groundstate_data
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

def validate_tolerance_options(value, _):

    valid_keys = ['max_iterations', 'energy', 'max_supercell', 'max_num_atoms']
    if value:
        tolerance_options = value.get_dict()
        for k, v in tolerance_options.items():
            if k == 'max_iterations' or k == 'max_num_atoms':
                try:
                    max_iterations = int(v)
                    if max_iterations < 1:
                        return f"The '{k}' value must be a positive integer."
                except ValueError:
                    return f"The '{k}' value must be an integer."
            elif k == 'energy':
                try:
                    energy_tolerance = float(v)
                    if energy_tolerance < 0:
                        return "The 'energy' value of the tolerance options must be nonnegative."
                except ValueError:
                    return "The 'energy' value of the tolerance options must be a number."
            elif k == 'max_supercell':
                try:
                    max_supercell = list(v)
                    if len(max_supercell) > 3 or not all(isinstance(number, int) for number in max_supercell):
                        return "The 'max_supercell' value should be a list with 3 integers."
                    elif any(number <= 0 for number in max_supercell):
                        return "The supercell dimensions should be positive integers."
                except TypeError:
                    return "The 'max_supercell' value should be a list with 3 integers."
            else:
                return f"Unrecognized option '{k}"


def validate_inputs(value, _):

    if 'structure' in value and 'tolerance_options' in value:
        structure = value['structure']
        tolerance_options = value['tolerance_options']
        if 'max_supercell' in tolerance_options: 
            if len(tolerance_options['max_supercell']) != len([pc for pc in structure.pc if pc]):
                return "The dimensions of the maximum allowed supercell are not compatible to the periodic boundary conditions of the structure."
        if 'max_num_atoms' in tolerance_options:
            if tolerance_options['max_num_atoms'] < len(structure.sites):
                return "The allowed maximum number of atoms should be greater or equal than the current number of atoms."

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
            validator=validate_tolerance_options,
            required=False,
            help='Energy and supercell tolerance values regarding the magnetic ground state search. For more informtaion, read the documentation.'
        )
        spec.input(
            'scale_processes_options',
            valid_type=orm.Dict,
            validator=validate_scale_processes_options,
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

        spec.inputs.validator = validate_inputs

        spec.output('exchange', valid_type=ExchangeData, required=True)
        spec.output('converged', valid_type=orm.Bool, required=True)

        spec.exit_code(300, 'ERROR_EXCHANGE_WC', message="The TB2JSiestaWorkChain failed.")

    def initialize(self):

        structure = self.inputs.structure
        num_atoms = len(structure.sites)
        
        tolerance = {
            'max_iterations': 3,
            'energy': 0.0,
            'max_supercell': [16 for pc in structure.pbc if pc],
            'max_num_atoms': 80 if num_atoms < 80 else 2*num_atoms
        }
        if 'tolerance_options' in self.inputs:
            tolerance.update(self.inputs.tolerance_options.get_dict())
        tolerance['max_iterations'] = int(tolerance['max_iterations'])
        tolerance['energy'] = float(tolerance['energy'])
        tolerance['max_supercell'] = list(tolerance['max_supercell'])
        
        self.ctx.structure = structure
        self.ctx.parameters = self.inputs.parameters
        self.ctx.kpoints = self.inputs.kpoints
        self.ctx.options = self.inputs.options

        self.ctx.coefficients = generate_coefficients(tolerance['max_supercell'])
        self.ctx.min_kpoint = np.ones(3)

        limits = np.ones(3)
        limits[list(structure.pbc)] = tolerance['max_supercell']
        self.ctx.limits = limits
        self.ctx.energy_tolerance = tolerance['energy']
        self.ctx.max_iterations = tolerance['max_iterations']
        self.ctx.max_num_atoms = tolerance['max_num_atoms']
        self.ctx.iterations = 0

    def minimum_not_gamma(self):

        cell = self.inputs.structure.cell
        supercell = self.ctx.structure.cell
        length_ratio  = np.linalg.norm(supercell, axis=1) / np.linalg.norm(cell, axis=1)
        if self.ctx.iterations == 0:
            magnorm = np.array([2.0])
        else:
            magmoms = self.ctx.magmoms
            magnorm = np.linalg.norm(magmoms, axis=-1)
        if magnorm.max() < 0.3:
            self.report(
                "The structure appears to be nonmagnetic"
            )
            return False
        elif (length_ratio > self.ctx.limits).any():
            self.report(
                "The resulting magnetic configuration exceeds the maximum allowed supercell size. Stopping the WorkChain..."
            )
            return False
        elif self.ctx.iterations >= self.ctx.max_iterations:
            self.report(
                "The magnetic groundstate has not been found in the maximum allowed iterations."
            )
            return False
        elif len(self.ctx.structure.sites) > self.ctx.max_num_atoms:
            self.report(
                "The resulting magnetic configuration exceeds the maximum allowed number of atoms. Stopping the WorkChain..."
            )
            return False

        return not ( self.ctx.min_kpoint == np.zeros(3) ).all()

    def calculate_Jiso(self):

        inputs = AttributeDict( self.exposed_inputs(TB2JSiestaWorkChain) )
        inputs['structure'] = self.ctx.structure
        inputs['parameters'] = self.ctx.parameters
        inputs['kpoints'] = self.ctx.kpoints
        inputs['options'] = self.ctx.options
        if self.ctx.iterations == 0:
            running = self.submit(MagneticOrientationWorkChain, **inputs)
            self.report(f"Launched MagneticOrientationWorkChain<{running.pk}> to optimize the unit cell magnetic configuration")
        else:
            inputs.pop('parent_calc_folder', None)
            running = self.submit(TB2JSiestaWorkChain, **inputs)
            self.report(f"Launched TB2JSiestaWorkChain<{running.pk}> to calculate the isotropic exchange constants")

        return ToContext(workchain_exchange=running)

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

        if self.ctx.iterations == 0:
            magmoms = exchange_data.magmoms()
            if exchange_data.non_collinear:
                self.ctx.magmoms = magmoms
            else:
                self.ctx.magmoms = np.zeros((len(magmoms), 3))
                self.ctx.magmoms[:, 2] = magmoms
            self.ctx.old_structure = exchange_data.get_structure()

        structure, parameters, q_vector = groundstate_data(
            exchange=exchange_data,
            parameters=self.inputs.parameters,
            magmoms=self.ctx.magmoms,
            tolerance=self.ctx.energy_tolerance,
            coefficients=self.ctx.coefficients,
            old_structure=self.ctx.old_structure
        )
        self.report(
            f"Iteration {self.ctx.iterations}, kpoint with minimum energy: {q_vector.round(4)}"
        )
        self.ctx.structure = structure
        self.ctx.parameters = parameters
        self.ctx.min_kpoint = q_vector

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
