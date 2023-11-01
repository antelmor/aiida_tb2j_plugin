from aiida import orm
from aiida.engine import WorkChain, ToContext, while_
from aiida.common import AttributeDict
from aiida_siesta.workflows.base import SiestaBaseWorkChain
import numpy as np

from ..utils import find_orientation, get_new_parameters
from ..data import ExchangeData
from . import TB2JSiestaWorkChain

class MagneticOrientationWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(TB2JSiestaWorkChain, exclude=('metadata') )

        spec.outline(
            cls.initialize,
            while_(cls.new_orientation_needed)(
                cls.optimize_geometry,
                cls.calculate_exchange,
                cls.analyze
            ),
            cls.return_results
        )

        spec.output('exchange', valid_type=ExchangeData, required=True)

        spec.exit_code(402, 'ERROR_EXCHANGE_WC', message="The TB2JSiestaWorkChain failed.")
        spec.exit_code(403, 'ERROR_SIESTA_GO', message="The TB2JSiestaWorkChain failed to optimize the geometry")

    def initialize(self):

        self.ctx.parameters = self.inputs.parameters
        self.ctx.magmoms_change = 2.0
        self.ctx.iteration = 0
        self.ctx.tb2j_keys = [
            'magnetic_elements', 'tb2j_code', 'tb2j_parameters', 'tb2j_options', 'converge_spin_orbit'
        ]

    def new_orientation_needed(self):

        return self.ctx.magmoms_change >= 1.0 and self.ctx.iteration < 3

    def optimize_geometry(self):

        param_dict = self.ctx.parameters.get_dict()
        param_dict['mdtypeofrun'] = 'CG'
        param_dict['mdsteps'] = 150
        param_dict['mdmaxforcetol'] = '0.01 eV/Ang'
        param_dict['mdvariablecell'] = True

        inputs = AttributeDict( self.exposed_inputs(TB2JSiestaWorkChain) )
        inputs['parameters'] = orm.Dict(dict=param_dict)
        for keyword in self.ctx.tb2j_keys:
            inputs.pop(keyword, None)
        if self.ctx.iteration != 0:
            inputs.pop('parent_calc_folder', None)
        running = self.submit(SiestaBaseWorkChain, **inputs)

        return ToContext(relax_run=running)

    def calculate_exchange(self):

        if not self.ctx.relax_run.is_finished_ok:
            return self.exit_codes.ERROR_SIESTA_GO

        structure = self.ctx.relax_run.outputs.output_structure
        parameters = self.ctx.parameters

        inputs = AttributeDict( self.exposed_inputs(TB2JSiestaWorkChain) )
        inputs['structure'] = structure
        inputs['parameters'] = parameters
        inputs.pop('parent_calc_folder', None)
        running = self.submit(TB2JSiestaWorkChain, **inputs)
        self.ctx.iteration += 1

        return ToContext(last_run=running)

    def analyze(self):

        if not self.ctx.last_run.is_finished_ok:
            return self.exit_codes.ERROR_EXCHANGE_WC

        exchange = self.ctx.last_run.outputs.exchange
        idx = sorted( set([pair[0] for pair in exchange.pairs]) )
        if exchange.non_collinear:
            magmoms = exchange.magmoms()
        else:
            magmoms = np.zeros((len(exchange.sites), 3))
            magmoms[:, 2] = exchange.magmoms()
        
        new_magmoms = find_orientation(exchange)
        self.ctx.magmoms_change = np.linalg.norm(new_magmoms - magmoms[idx], axis=-1).max()
        magmoms[idx] = new_magmoms
        new_parameters = get_new_parameters(magmoms, self.inputs.parameters)
        self.ctx.parameters = new_parameters

    def return_results(self):

        from aiida.engine import ExitCode

        if not self.ctx.last_run.is_finished_ok:
            return self.exit_codes.ERROR_EXCHANGE_WC

        exchange_data = self.ctx.last_run.outputs.exchange
        self.out('exchange', exchange_data)
        
        self.report('MagneticOrientation workchain finished succesfully')

        return ExitCode(0)
