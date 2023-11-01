from aiida.parsers import Parser
from aiida.common import exceptions
from ..data import CurieData
import numpy as np

class VampireParser(Parser):

    def parse(self, **kwargs):

        from aiida.engine import ExitCode

        try:
            retrieved = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED

        if 'output' not in retrieved.list_object_names():
            return self.exit_codes.ERROR_OUTPUT_VAMPIRE_MISSING

        try:
            curie_data = self._set_curie_data()
        except (IOError, OSError):
            self.exit_codes.ERROR_OUTPUT_CURIE_DATA

        self.out('output_data', curie_data)

        parser_info = {}
        parser_info['parser_info'] = 'AiiDA Vampire Parser'
        parser_info['parser_warnings'] = []
        parser_info['output_data_filename'] = 'output'

        return ExitCode(0)


    def _set_curie_data(self):

        curie_data = CurieData()
        with self.retrieved.base.repository.open('output', 'r') as File:
            output = np.loadtxt(File)

        output_options = self.node.inputs['output_options']
        exchange = self.node.inputs.exchange
        idx = sorted( set([pair[0] for pair in exchange.pairs]) )
        magmoms = exchange.magmoms().round(2)
        unique_magmoms = np.array(list( set(tuple(a) for a in magmoms[idx]) ))
        num_materials = len(unique_magmoms)
        try:
            ti = output_options.index('temperature')
            mi = output_options.index('material-mean-magnetisation-length')
            curie_data.set_temperature_values(output[:, ti])
            curie_data.set_magnetization_values(output[:, mi:mi+num_materials])
        except ValueError:
            pass

        curie_data.set_array('output', output)

        return curie_data
