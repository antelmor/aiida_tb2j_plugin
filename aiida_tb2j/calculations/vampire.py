from aiida.engine import CalcJob
from aiida import orm
from aiida.common import CalcInfo, CodeInfo
from ..data import ExchangeData, CurieData
from ..utils import write_vampire_files
import numpy as np

class VampireCalculation(CalcJob):

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input(
            'code',
            valid_type=orm.Code,
            required=True,
            help="Vampire code."
        )
        spec.input(
            'exchange',
            valid_type=ExchangeData,
            required=True,
            help="ExchangeData containing the structure and exchange interactions information."
        )
        spec.input(
            'isotropic',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="Whether or not to use only the isotropic exchange constants."
        )
        spec.input(
            'max_distance',
            valid_type=orm.Float,
            required=False,
            help="Only the exchange interactions with a corresponding distance smaller than this value will be considered."
        )
        spec.input(
            'parameters',
            valid_type=orm.Dict,
            required=True,
            help="Input parameters of the Vampire calculation."
        )
        spec.input(
            'output_options',
            valid_type=orm.List,
            required=True,
            help="Comands that control what data is output from the calculation."
        )

        spec.inputs['metadata']['options']['parser_name'].default = 'vampire.parser'

        spec.output('output_data', valid_type=CurieData, required=True, help='Contains the results arrays')
        spec.exit_code(400, 'ERROR_NO_RETRIEVED', message="The retrieved folder data node could not be accessed.")
        spec.exit_code(401, 'ERROR_OUTPUT_VAMPIRE_MISSING', message="The retrieved folder does not contain the 'output' file.")
        spec.exit_code(402, 'ERROR_OUTPUT_VAMPIRE_READ', message="The 'output' file can not be read.")
        spec.exit_code(403, 'ERROR_OUTPUT_CURIE_DATA', message="The CurieData object could not be created.")

    def prepare_for_submission(self, folder):

        code = self.inputs.code

        exchange = self.inputs.exchange
        pbc = exchange.pbc
        lattice_parameters = np.linalg.norm(exchange.cell, axis=-1)
        coordinates = ['x', 'y', 'z']
        inputs_dict = {'create:full': None}
        inputs_dict.update({f'create:periodic-boundaries-{coordinates[i]}': None for i in range(3) if pbc[i]})
        inputs_dict.update({'material:file': '"vampire.mat"', 'material:unit-cell-file': '"vampire.UCF"'})
        inputs_dict.update({f'dimensions:unit-cell-size-{coordinates[i]}': lattice_parameters[i] for i in range(3)})
        inputs_dict.update({f'dimensions:system-size-{coordinates[i]}': '25.0 !nm' if pbc[i] else f'{(0.1*lattice_parameters[i]).round(2)} !nm' for i in range(3)})

        param_dict = self.inputs.parameters.get_dict()
        inputs_dict.update(param_dict)

        input_filename = folder.get_abs_path('input')
        with open(input_filename, 'w', encoding='utf8') as File:
            for key, value in inputs_dict.items():
                if value is None:
                    line = f'{key}\n'
                else:
                    line = f'{key} = {value}\n'
                File.write(line)
            for option in self.inputs.output_options:
                File.write(f'output:{option}\n')

        if 'max_distance' not in self.inputs:
            max_distance = np.inf
        else:
            max_distance = self.inputs.max_distance.value
        ucf_filename = folder.get_abs_path('vampire.UCF')
        mat_filename = folder.get_abs_path('vampire.mat')
        write_vampire_files(
            exchange, 
            material_filename=mat_filename, 
            UCF_filename=ucf_filename,
            isotropic=self.inputs.isotropic.value,
            max_distance=max_distance
        )

        codeinfo = CodeInfo()
        codeinfo.code_uuid = code.uuid
        
        calcinfo = CalcInfo()
        calcinfo.uuid = str(self.uuid)
        calcinfo.local_copy_list = []
        calcinfo.remote_copy_list = []
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = ['output']

        return calcinfo
