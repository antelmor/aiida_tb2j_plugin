from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, if_, while_
from aiida_siesta.workflows.base import SiestaBaseWorkChain as ParentSiestaBaseWorkChain, validate_inputs
from aiida_siesta.utils.tkdict import FDFDict
from ..calculations import TB2JCalculation, SiestaCalculation
from ..calculations.tb2j import validate_parameters as validate_tb2j_parameters
from ..data import ExchangeData

def validate_siesta_parameters(value, _):

    if value:
        from aiida_siesta.calculations.siesta import validate_parameters
        message = validate_parameters(value, _)
        if message is not None:
            return message

        from aiida_siesta.utils.tkdict import FDFDict
        input_params = FDFDict(value.get_dict())
        sortedkey = sorted(input_params.get_filtered_items())
        spin = False
        newsintax = False
        oldsintax = False
        ncdf_parameters = ['cdfsave', 'savehs', 'writedmhsnetcdf']

        for k, v in sortedkey:
            if k == "spinpolarized":
                if v in [True, 'T', 'true', '.true.']:
                    oldsintax = True
                    spin = True
            elif k == "spin":
                if v in ['polarized', 'non-collinear', 'spin-orbit']:
                    newsintax = True
                    spin = True

        if newsintax and oldsintax:
            import warnings
            warnings.warn(("WARNING: in the siesta input parameters, both keywork 'spin' and " +
                "'spinpolarized' have been detected. This might confuse the WorkChain and return " +
                "unexpected outputs")
            )

        if not spin:
            return ("The exchange coefficients calculation requires a spin siesta calculation. " +
                "Set spin to either polarized, noncollinear or spinorbit.")

        missing_ncdf = ncdf_parameters
        for k, v in sortedkey:
            if k in ncdf_parameters and v in [True, 'T', 'true', '.true.']:
                missing_ncdf.remove(k)
        if missing_ncdf:
            return (f"The options {str(missing_ncdf).strip('[]')} need to be set up to True " +
                "in order to use the TB2J code.")

def validate_symbols(value, _):

    if value:
        is_valid_symbol = orm.nodes.data.structure.is_valid_symbol
        if len(value) == 0:
            return "The list of elements cannot be empty."
        elif not all(is_valid_symbol(sym) for sym in value):
            return (f"At least one element of the symbol list {list(value)} has not " +
                "been recognized")

def validate_siesta_inputs(value, _):

    message = validate_inputs(value, _)
    if message:
        return message
    if 'spin_angles' in value and 'parent_calc_folder' not in value:
        return "If 'spin_angles' are specified, a 'parent_calc_folder' must be provided."

class SiestaBaseWorkChain(ParentSiestaBaseWorkChain):

    _process_class = SiestaCalculation
    _proc_exit_cod = _process_class.exit_codes

    @classmethod
    def define(cls, spec):

        super().define(spec)
        spec.expose_inputs(SiestaCalculation, include=('spin_angles',))

        spec.outline(
            cls.setup,
            cls.preprocess,
            while_(cls.should_run_process)(
                cls.prepare_inputs,
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
            cls.postprocess,
        )

        spec.expose_outputs(SiestaCalculation)
        spec.inputs.validator = validate_siesta_inputs

    def preprocess(self):

        super().prepare_inputs()

    def prepare_inputs(self):

        if self.ctx.iteration == 0 and 'spin_angles' in self.inputs:
            self.ctx.inputs['spin_angles'] = self.inputs.spin_angles
            

class TB2JSiestaWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        
        super().define(spec)
        spec.expose_inputs(SiestaBaseWorkChain, exclude=('metadata', 'parameters'))
        
        spec.input(
            'parameters',
            valid_type=orm.Dict,
            validator=validate_siesta_parameters,
            help='Input parameters for the SIESTA code'
        ) 
        spec.input(
            'magnetic_elements',
            valid_type=orm.List,
            required=False,
            validator=validate_symbols,
            help='Elements for which the exchange interaction is considered.'
                'If not provided, the code selects those with d or f orbitals.'
        )
        spec.input(
            'tb2j_code',
            valid_type=orm.Code,
            help='TB2J Siesta binary.'
        )
        spec.input(
            'tb2j_parameters', 
            valid_type=orm.Dict, 
            required=False,
            validator=validate_tb2j_parameters,
            help='Parameters used by the TB2J code'
        )
        spec.input(
            'tb2j_options', 
            valid_type=orm.Dict, 
            required=False, 
            help='TB2J code resources and options.'
        )
        spec.input(
            'converge_spin_orbit',
            valid_type=orm.Bool,
            default=lambda : orm.Bool(False),
            help='Wether perform a non-collinear calculation before spin-orbit'
        )

        spec.outline(
            if_(cls.converge_before)(
                cls.run_siesta_prior
            ),
            cls.run_siesta_wc,
            cls.run_tb2j,
            cls.return_results
        )

        spec.output('exchange', valid_type=ExchangeData, required=False)
        spec.output('retrieved', valid_type=orm.FolderData)
        spec.output('remote_folder', valid_type=orm.RemoteData)

        spec.exit_code(400, 'ERROR_SIESTA_WC', message='The main SiestaBaseWorkChain failed.')
        spec.exit_code(401, 'ERROR_TB2J_PLUGIN', message='The TB2J calculation failed.')
        spec.exit_code(403, 'ERROR_SIESTA_AUX', message='The support SiestaBaseWorkChain failed.')

    def converge_before(self):

        return self.inputs.converge_spin_orbit

    def run_siesta_prior(self):

        param_dict = self.inputs.parameters.get_dict()
        param_dict['spin'] = 'non-collinear'
        inputs = AttributeDict(self.exposed_inputs(SiestaBaseWorkChain))
        inputs['parameters'] = orm.Dict(dict=param_dict)
        running = self.submit(SiestaBaseWorkChain, **inputs)
        self.report(
            f"Launched SiestaBaseWorkChain<{running.pk}> to help convergence."
        )

        return ToContext(siesta_prior=running)

    def run_siesta_wc(self):

        inputs = AttributeDict(self.exposed_inputs(SiestaBaseWorkChain))
        inputs['parameters'] = self.inputs.parameters
        if self.inputs.converge_spin_orbit:
            if not self.ctx.siesta_prior.is_finished_ok:
                return self.exit_codes.ERROR_SIESTA_AUX
            inputs.pop('spin_angles', None)
            inputs['parent_calc_folder'] = self.ctx.siesta_prior.outputs.remote_folder

        running = self.submit(SiestaBaseWorkChain, **inputs)
        self.report(
            f"Launched SiestaBaseWorkChain<{running.pk}> to perform the Siesta calculation."
        )

        return ToContext(siesta_wc=running)

    def _magnetic_elements(self):

        from ..utils import get_magnetic_elements

        try:
            magnetic_elements = self.inputs.magnetic_elements
        except AttributeError:
            if 'pseudos' in self.inputs:
                pseudos = self.inputs.pseudos
            else:
                pseudo_group = orm.Group.get(label=self.inputs.pseudo_family.value)
                pseudos = pseudo_group.get_pseudos(structure=self.inputs.structure)
            magnetic_elements = get_magnetic_elements(pseudos)

        if not magnetic_elements:
            magnetic_elements = orm.List(list=[pseudo for pseudo in pseudos])
            self.report(
                "WARNING: The structure might not be magnetic."
            )

        return magnetic_elements

    def run_tb2j(self):

        if not self.ctx.siesta_wc.is_finished_ok:
            return self.exit_codes.ERROR_SIESTA_WC
 
        magnetic_elements = self._magnetic_elements()

        tb2j_inputs = {
            'code': self.inputs.tb2j_code,
            'elements': magnetic_elements,
            'structure': self.inputs.structure
        }

        if 'tb2j_parameters' in self.inputs:
            tb2j_inputs['parameters'] = self.inputs.tb2j_parameters
            if 'kmesh' in self.inputs.tb2j_parameters:
                del tb2j_inputs['structure']

        if 'tb2j_options' in self.inputs:
            optio = self.inputs.tb2j_options.get_dict()
        else:
            optio = self.inputs.options.get_dict()
            optio['resources']['num_machines'] = 1
            optio['resources']['num_cores_per_machine'] = 1 
            optio['withmpi'] = False

        siesta_remote = self.ctx.siesta_wc.outputs.remote_folder
        tb2j_inputs['siesta_remote'] = siesta_remote
        tb2j_inputs['metadata'] = {'options': optio}
        running = self.submit(TB2JCalculation, **tb2j_inputs)
        self.report(f'Launching TB2JCalculation<{running.pk}>')            
       
        return ToContext(tb2j_step = running)

    def return_results(self):

        from aiida.engine import ExitCode

        if not self.ctx.tb2j_step.is_finished_ok:
            return self.exit_codes.ERROR_TB2J_PLUGIN

        if 'exchange' in self.ctx.tb2j_step.outputs:
            exchange_data = self.ctx.tb2j_step.outputs.exchange
            self.out('exchange', exchange_data)
        
        retrieved = self.ctx.tb2j_step.outputs.retrieved
        remote = self.ctx.tb2j_step.outputs.remote_folder
        self.out('retrieved', retrieved)
        self.out('remote_folder', remote)

        self.report('TB2J workchain completed succesfully.')
        return ExitCode(0)
