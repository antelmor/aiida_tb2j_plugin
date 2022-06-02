from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext
from aiida_siesta.workflows.base import SiestaBaseWorkChain
from aiida_siesta.utils.tkdict import FDFDict
from ..calculations import TB2JCalculation
from ..calculations.tb2j import validate_parameters as validate_tb2j_parameters
from ..data import ExchangeData

class TB2JSiestaWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(SiestaBaseWorkChain, exclude=('metadata',))
        
        spec.input(
            'tb2j_code', 
            valid_type=orm.Code, 
            help='TB2J siesta binary.'
        )
        spec.input(
            'magnetic_elements',
            valid_type=orm.List,
            required=False,
            validator=orm.nodes.data.structure.validate_symbols_tuple,
            help='Elements for which the exchange interaction is considered.'
                'If not provided, the code selects those with d or f orbitals.'
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

        spec.outline(
            cls.checks,
            cls.run_siesta_wc,
            cls.run_tb2j,
            cls.return_results
        )

        spec.output('exchange', valid_type=ExchangeData, required=False)
        spec.output('remote_folder', valid_type=orm.RemoteData)

        spec.exit_code(200, 'ERROR_BASE_WC', message='The main SiestaBaseWorkChain failed.')
        spec.exit_code(201, 'ERROR_TB2J_PLUGIN', message='The TB2J calculation failed.')

    def checks(self):

        code = self.inputs.code
        param_dict = self.inputs.parameters.get_dict()
        translatedkey = FDFDict(param_dict)
        sortedkey = sorted(translatedkey.get_filtered_items())
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
            self.report(
                "WARNING: in the siesta input parameters, both keywork 'spin' and "
                "'spinpolarized' have been detected. This might confuse the WorkChain and return "
                "unexpected outputs"
            )

        if not spin:
            raise ValueError(
                "The exchange coefficients calculation requires a spin siesta calculation. "
                "Set spin to either polarized, noncollinear or spinorbit."
            ) 

        missing_ncdf = ncdf_parameters
        for k, v in sortedkey:
            if k in ncdf_parameters and v in [True, 'T', 'true', '.true.']:
                missing_ncdf.remove(k)
        if missing_ncdf:
            raise ValueError(
                f"The options {str(missing_ncdf).strip('[]')} need to be set up to True"
                "in order to use the TB2J code."
            )

    def run_siesta_wc(self):

        inputs = AttributeDict(self.exposed_inputs(SiestaBaseWorkChain))        
        running = self.submit(SiestaBaseWorkChain, **inputs)
        self.report(
            f"Launched SiestaBaseWorkChain<{running.pk}> to perform the siesta calculation."
        )
        return ToContext(workchain_base=running)

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

        if not self.ctx.workchain_base.is_finished_ok:
            return self.exit_codes.ERROR_BASE_WC
 
        magnetic_elements = self._magnetic_elements()

        tb2j_inputs = {
            'code': self.inputs.tb2j_code,
            'elements': magnetic_elements
        }
        if 'tb2j_parameters' in self.inputs:
            tb2j_inputs['parameters'] = self.inputs.tb2j_parameters
        if 'tb2j_options' in self.inputs:
            optio = self.inputs.tb2j_options.get_dict()
        else:
            optio = self.inputs.options.get_dict()
            optio['resources']['num_machines'] = 1
            optio['resources']['num_cores_per_machine'] = 1 
        optio['withmpi'] = False

        siesta_remote = self.ctx.workchain_base.outputs.remote_folder
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
        
        remote_folder = self.ctx.tb2j_step.outputs.remote_folder
        self.out('remote_folder', remote_folder)

        self.report('TB2J workchain completed succesfully.')
        return ExitCode(0)
