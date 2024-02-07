from aiida import orm
from aiida.engine import WorkChain, while_, ToContext, calcfunction
from aiida.common import AttributeDict

from . import TB2JSiestaWorkChain, DMIWorkChain 
from ..data import ExchangeData
from ..utils import groundstate_data
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

def validate_optimization_options(value, _):

    if value:
        optimization_options = value.get_dict()
        for k, v in optimization_options.items():
            if k == 'max_iterations' or k == 'max_num_atoms':
                try:
                    max_iterations = int(v)
                    if max_iterations < 1:
                        return f"The '{k}' value must be a positive integer."
                except ValueError:
                    return f"The '{k}' value must be an integer."
            elif k == 'energy' or k == 'min_magmom':
                try:
                    threshold = float(v)
                    if threshold < 0:
                        return f"The '{k}' value of the tolerance options must be nonnegative."
                except ValueError:
                    return "The '{k}' value of the tolerance options must be a number."
            elif k == 'isotropic':
                if not isinstance(v, bool):
                    return "The value of the argument 'isotropic' must be of type 'bool'."
            else:
                return f"Unrecognized option '{k}"

def validate_inputs(value, _):

    if 'structure' in value and 'optimization_options' in value:
        structure = value['structure']
        optimization_options = value['optimization_options']
        if 'max_num_atoms' in optimization_options:
            if optimization_options['max_num_atoms'] < len(structure.sites):
                return "The allowed maximum number of atoms should be greater or equal than the current number of atoms."

@calcfunction
def set_groundstate_info(
        exchange: ExchangeData,
        optimization_options: orm.Dict
    ) -> orm.Dict:

    from ..utils import find_minimum_kpoints

    optimize_dict = optimization_options.get_dict()
    threshold = optimize_dict.pop('energy', 1e-4)
    
    min_kpoint = find_minimum_kpoints(exchange, threshold=threshold)
    min_energy = exchange._magnon_energies(min_kpoint.reshape(-1, 3)).min()
    
    groundstate_found = (min_kpoint == np.zeros(3)).all() and min_energy >= -threshold

    return orm.Dict({
        'min_kpoint': min_kpoint.tolist(),
        'min_energy': min_energy,
        'groundstate': bool(groundstate_found)
    })

class GroundStateWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(TB2JSiestaWorkChain, exclude=('metadata',))

        spec.input(
            'optimization_options',
            valid_type=orm.Dict,
            validator=validate_optimization_options,
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
            while_(cls.groundstate_not_found)(
                cls.calculate_exchange,
                cls.analyze,
            ),
            cls.return_results
        )

        spec.inputs.validator = validate_inputs

        spec.output('exchange', valid_type=ExchangeData, required=True)
        spec.output('groundstate_info', valid_type=orm.Dict, required=True)

        spec.exit_code(300, 'ERROR_EXCHANGE_WC', message="The WorkChain to calculate the exchange parameters failed.")

    def initialize(self):

        structure = self.inputs.structure
        num_atoms = len(structure.sites)
         
        optimization_options = {}
        if 'optimization_options' in self.inputs:
            optimization_options.update(self.inputs.optimization_options.get_dict())
        
        self.ctx.structure = structure
        self.ctx.parameters = self.inputs.parameters
        self.ctx.kpoints = self.inputs.kpoints
        self.ctx.options = self.inputs.options

        self.ctx.isotropic = optimization_options.pop('isotropic', True)
        self.ctx.energy_threshold = float(optimization_options.pop('energy', 1e-4))
        self.ctx.max_iterations = int(optimization_options.pop('max_iterations', 3))
        self.ctx.max_num_atoms = int(optimization_options.pop('max_num_atoms', 80 if num_atoms < 80 else 2*num_atoms))
        self.ctx.min_magmom = float(optimization_options.pop('min_magmom', 0.1))

        self.ctx.magmoms = np.ones(3)
        self.ctx.min_kpoint = np.ones(3)
        self.ctx.min_energy = -1
        self.ctx.iterations = 0

    def groundstate_not_found(self):

        magnorm = np.linalg.norm(self.ctx.magmoms, axis=-1)
        if magnorm.max() < self.ctx.min_magmom:
            self.report(
                "The structure appears to be nonmagnetic"
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

        groundstate_found = (self.ctx.min_kpoint == np.zeros(3)).all() and self.ctx.min_energy >= -self.ctx.energy_threshold

        return not groundstate_found

    def calculate_exchange(self):

        inputs = AttributeDict( self.exposed_inputs(TB2JSiestaWorkChain) )
        inputs['structure'] = self.ctx.structure
        inputs['parameters'] = self.ctx.parameters
        inputs['kpoints'] = self.ctx.kpoints
        inputs['options'] = self.ctx.options
            
        inputs.pop('parent_calc_folder', None)
        if self.ctx.isotropic:
            running = self.submit(TB2JSiestaWorkChain, **inputs)
            self.report(f"Launched TB2JSiestaWorkChain<{running.pk}> to calculate the isotropic exchange constants")
        else:
            running = self.submit(DMIWorkChain, **inputs)
            self.report(f"Launched DMIWorkChain<{running.pk}> to calculate the excange parameters.")

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

        structure, parameters, q_vector = groundstate_data(
            exchange_data,
            self.inputs.parameters,
            optimize_magmoms=True,
            optimizer_kwargs={'threshold': self.ctx.energy_threshold}
        ) 

        self.ctx.min_energy = exchange_data._magnon_energies(q_vector.reshape(-1, 3)).min()
        self.ctx.structure = structure
        self.ctx.parameters = parameters
        self.ctx.min_kpoint = q_vector
        self.report(
            f"Iteration {self.ctx.iterations}, kpoint <{q_vector.round(4)}> with minimum energy: {self.ctx.min_energy}"
        )

        new_kpoints = self._get_new_kpoints()
        self.ctx.kpoints = new_kpoints
        if 'scale_processes_options' in self.inputs:
            new_options = self._get_new_options()
            self.ctx.options = new_options
        self.ctx.iterations += 1

    def return_results(self):

        from aiida.engine import ExitCode

        final_exchange = self.ctx.workchain_exchange.outputs.exchange
        opt_options = self.inputs.optimization_options if 'optimization_options' in self.inputs else self.inputs.options
        groundstate_info = set_groundstate_info(final_exchange, opt_options)

        self.out('exchange', final_exchange)
        self.out('groundstate_info', groundstate_info)

        self.report('Magnetic GroundState workchain completed.')
        return ExitCode(0) 
