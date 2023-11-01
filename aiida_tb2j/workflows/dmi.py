from aiida import orm
from aiida.common import AttributeDict
from aiida.common.exceptions import NotExistentAttributeError
from aiida.engine import calcfunction, ToContext, WorkChain
from .siesta import TB2JSiestaWorkChain
from ..data import ExchangeData

def validate_noncollinear_parameters(value, _):

    if value:
        from .siesta import validate_siesta_parameters
        message = validate_siesta_parameters(value, _)
        if message is not None:
            return message

        from aiida_siesta.utils.tkdict import FDFDict
        input_params = FDFDict(value.get_dict())
        sortedkey = sorted(input_params.get_filtered_items())
        collinear = True
        oldsintax = False

        for k, v in sortedkey:
            if k == "spinpolarized":
                if v in [True, 'T', 'true', '.true.']:
                    oldsintax = True
            elif k == "spin":
                if v in ['non-collinear', 'spin-orbit']:
                    collinear = False

        if oldsintax:
            for k, v in sortedkey:
                if k == "noncollinearspin":
                    if v in [True, 'T', 'true', '.true.']:
                        collinear = False

        if collinear:
            return ("The DMI work chain requires non-collinear spin. Set the 'spin' parameter " +
                "to either 'non-collinear' or 'spin-orbit'.")

def validate_spin_mode(value, _):

    if value:
        if value not in ['collinear', 'non-collinear', 'long']:
            return (f"Unrecognized option '{value}'. Available options are: 'collinear', 'non-collinear', 'long'.")

def validate_rotation_mode(value, _):

    if value:
        if value not in ['structure', 'spin']:
            return f"Unrecognized option '{value}'. Available options are: 'structure' and 'spin'." 

def get_rotated_structures(structure, mode='collinear'):

    from copy import deepcopy

    atoms = structure.get_ase()
    if mode == 'collinear':
        rotation_axes = ['x', 'y']
    elif mode == 'non-collinear':
        rotation_axes = [
            (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)
        ]
    else:
        rotation_axes = [
            (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1),
            (-1, 1, 0), (-1, 0, 1), (0, -1, 1)
        ]

    structures = {}
    for axis in rotation_axes:
        rotated_atoms = deepcopy(atoms)
        rotated_atoms.rotate(90, axis, rotate_cell=True)
        rotated_structure = orm.StructureData(ase=rotated_atoms)
        structures[axis] = rotated_structure

    return structures

def get_rotation_angles(mode='collinear'):

    angles =  {
        'x': orm.List([0.0, 90.0, 0.0]), 
        'y': orm.List([-90.0, 0.0, 0.0])
    }
    if mode != 'collinear':
        angles.update({
            'zx': orm.List([0.0, 45.0, 0.0]),
            'xy': orm.List([0.0, 90.0, 45.0]),
            'yz': orm.List([0.0, 45.0, 90.0])
        })

    return angles

def get_siesta_remote(process):

    siesta_nodes = [node for node in process.called if node.process_label == 'SiestaBaseWorkChain']
    siesta_nodes.sort(key=lambda node : node.pk)

    return siesta_nodes[0].outputs.remote_folder

def choose_main_process(*args):

    def get_energy(process):
        natoms = len(process.inputs.structure.sites)
        for node in process.called:
            try:
                energy = node.outputs.output_parameters['E_KS']
            except NotExistentAttributeError:
                continue
        return round(energy / natoms, 3)
    procs = sorted(args, key=get_energy)

    return procs[0]

@calcfunction
def get_exchange(**kwargs):

    from ..utils import Merger
    from ..parsers import correct_content

    if 'structure' in kwargs:
        structure = kwargs.pop('structure')
        pbc = structure.pbc
    else:
        pbc= (True, True, True)
 
    main_folder = kwargs.pop('main_folder', None)

    folders = kwargs.values()
    merger = Merger(*folders, main_folder=main_folder)
    merger.merge_Jiso()
    merger.merge_Jani()
    merger.merge_DMI()

    content = merger.main_dat.__dict__
    correct_content(content)

    exchange_data = ExchangeData.load_tb2j_content(content, pbc=pbc, isotropic=False)

    return exchange_data

class DMIWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(TB2JSiestaWorkChain, exclude=('metadata', 'parameters'))

        spec.input(
            'parameters',
            valid_type=orm.Dict,
            help='Input parameters for the SIESTA code',
            validator=validate_noncollinear_parameters,
            required=True
        )
        spec.input(
            'spin_mode',
            valid_type=orm.Str,
            default=lambda: orm.Str('non-collinear'),
            help='Specifies the number of rotations needed.',
            validator=validate_spin_mode,
            required=True
        )
        spec.input(
            'rotation_mode',
            valid_type=orm.Str,
            default=lambda: orm.Str('structure'),
            help='Specifies wheter the structure or the spins will be rotated.',
            validator=validate_rotation_mode,
            required=True
        )

        spec.outline(
            cls.run_reference_process,
            cls.run_rotated_processes,
            cls.return_results
        )

        spec.output('exchange', valid_type=ExchangeData, required=True)

        spec.exit_code(400, 'ERROR_TB2J_REF', message='The reference TB2J workflows failed.')
        spec.exit_code(401, 'ERROR_TB2J_ROT', message='At least one of the rotated TB2J workflows failed.')
        spec.exit_code(402, 'ERROR_MERGE_TB2J', message='The merging of the exchange values failed.')

    def run_reference_process(self):

        inputs = self.exposed_inputs(TB2JSiestaWorkChain)

        if 'tb2j_options' in self.inputs:
            optio = self.inputs.tb2j_options.get_dict()
            del inputs['tb2j_options']
        else:
            optio = self.inputs.options.get_dict()
            optio['resources']['num_machines'] = 1
            optio['resources']['num_cores_per_machine'] = 1
        optio['parser_name'] = 'tb2j.basic'
        optio['withmpi'] = False
        inputs['tb2j_options'] = orm.Dict( dict=optio )
        inputs['parameters'] = self.inputs.parameters

        self.ctx.inputs = inputs
        running = self.submit(TB2JSiestaWorkChain, **inputs)
        self.report(
            f"Launched TB2JSiestaWorkChain<{running.pk}> for the reference structure."
        )

        return ToContext(ref_process=running)

    def run_rotated_processes(self):

        ref_process = self.ctx.ref_process
        if not ref_process.is_finished_ok:
            return self.exit_codes.ERROR_TB2J_REF

        inputs = self.ctx.inputs
        if self.inputs.rotation_mode == 'structure':
            del inputs['structure']
            inputs.pop('parent_calc_folder', None)
            structures = get_rotated_structures(self.inputs.structure, mode=self.inputs.spin_mode)
            for key, structure in structures.items():
                process = self.submit(TB2JSiestaWorkChain, **inputs, structure=structure)
                self.to_context(**{f'tb2j_{key}': process})
                self.report(
                    f"Launched TB2JSiestaWorkChain<{process.pk}> for the {key}-rotated structure."
                )
            axes = structures.keys()
        elif self.inputs.rotation_mode == 'spin':
            inputs['parent_calc_folder'] = get_siesta_remote(ref_process)
            spin_angles = get_rotation_angles(self.inputs.spin_mode)
            for key, angles in spin_angles.items():
                process = self.submit(TB2JSiestaWorkChain, **inputs, spin_angles=angles)
                self.to_context(**{f'tb2j_{key}': process})
                self.report(
                    f"Launched TB2JSiestaWorkChain<{process.pk}> for the {key}-rotated calculation."
                )
            axes = spin_angles.keys()

        self.ctx.rotation_axes = list(axes)

    def return_results(self):

        from aiida.engine import ExitCode

        ref_process = self.ctx.ref_process
        rotation_axes = self.ctx.rotation_axes
        rotated_processes = {axis: self.ctx[f'tb2j_{axis}'] for axis in rotation_axes}
        if not all([process.is_finished_ok for process in list(rotated_processes.values()) + [ref_process,]]):
            return self.exit_codes.ERROR_TB2J_ROT
        main_process = choose_main_process(ref_process, *rotated_processes.values())

        folders = {f'folder_{rotation_axes.index(axis)}': process.outputs.retrieved for axis, process in rotated_processes.items()}
        folders['ref_folder'] = ref_process.outputs.retrieved
        folders['main_folder'] = main_process.outputs.retrieved
        folders['structure'] = self.inputs.structure
    
        exchange = get_exchange(**folders)

        if 'array|Jiso' in exchange.attributes and 'array|DMI' in exchange.attributes:
            self.report(
                "The exchange constants including the DMI interaction and the " 
                "anisotropic exchange were obtained succesfully."
            )
        else:
            return self.exit_codes.ERROR_MERGE_TB2J

        self.out('exchange', exchange)

        return ExitCode(0)
