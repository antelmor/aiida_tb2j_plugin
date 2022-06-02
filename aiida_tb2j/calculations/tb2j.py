import os
from aiida.engine import CalcJob
from aiida import orm
from aiida.common import CalcInfo, CodeInfo
from ..data import ExchangeData

def validate_parameters(value, _):

    floattype_keys = [
        'rcut',
        'efermi',
        'emin',
        'emax',
        'cutoff',
    ]
    booltype_keys = [
        'use_cache',
        'orb_decomposition',
    ]
    inttype_keys = [
        'nz',
        'exclude_orbs'
    ]
    unvalid_keys = [
        'fdf_fname',
        'elements',
        'np',
        'output_path'
    ]

    if value:
        params = value.get_dict()
        for key, value in params.items():
            if key in floattype_keys:
                if not isinstance(value, (float, int)):
                    return f"'{key}' option must be a real number."
            elif key in booltype_keys:
                if not isinstance(value, bool):
                    return f"'{key}' option must be a boolean."
            elif key in inttype_keys:
                if not isinstance(value, int):
                    return f"'{key}' option must be an integer."
            elif key in unvalid_keys:
                return f"You can't specify the '{key}' option in the parameters input port."
            elif key == 'kmesh':
                try:
                    kmesh = list(value)
                    if len(kmesh) != 3 or not all(isinstance(x, int) for x in kmesh):
                        return "The 'kmesh' option must contain 3 integers."
                except TypeError:
                    return "The 'kmesh' option must be an iterable."
            elif key == 'magnetic_elements':
                pass
            else:
                return f"Unrecognized option '{key}'"               

class TB2JCalculation(CalcJob):

    _restart_copy_from = os.path.join('./', '*.nc')
    _restart_copy_to = './'

    @classmethod
    def define(cls, spec):
        super(TB2JCalculation, cls).define(spec)

        spec.input(
            'code', 
            valid_type=orm.Code, 
            required=True, 
            help='TB2J siesta binary.'
        )
        spec.input(
            'siesta_remote', 
            valid_type=orm.RemoteData, 
            required=True, 
            help='Parent remote folder.'
        )
        spec.input(
            'elements', 
            valid_type=orm.List, 
            required=True,
            validator=orm.nodes.data.structure.validate_symbols_tuple, 
            help='List of magnetic elements.'
        )
        spec.input(
            'parameters', 
            valid_type=orm.Dict,
            required=False,
            validator=validate_parameters,
            help='Contains all the optional parameters for the TB2J calculation.'
        ) 

        spec.inputs['metadata']['options']['parser_name'].default = 'tb2j.parser'

        spec.output('exchange', valid_type=ExchangeData, required=False, help='Exchange interaction data for different magnetic atom pairs.')

        spec.exit_code(100, 'ERROR_NO_RETRIEVED_FOLDER', message="The retrieved folder data node could not be accessed.")
        spec.exit_code(101, 'ERROR_OUTPUT_PICKLE_MISSING', message="The retrieved folder does not contain the 'TB2J.pickle' file.")
        spec.exit_code(102, 'ERROR_OUTPUT_PICKLE_READ', message="The 'TB2J.pickle' file can not be read.")
        spec.exit_code(103, 'ERROR_OUTPUT_EXCHANGE_DATA', message="The ExchangeData object could not be created.")
 
    def prepare_for_submission(self, folder):

        code = self.inputs.code
        siesta_remote = self.inputs.siesta_remote
        siesta_path = siesta_remote.get_remote_path()
        fdf_fname = 'aiida.fdf'
        kmesh = [5, 5, 5]
        param_dict = {}
        np = self.inputs.metadata.options.resources['num_cores_per_machine']

        if 'parameters' in self.inputs:
            param_dict = self.inputs.parameters.get_dict()
            try:
                kmesh = param_dict.pop('kmesh')
            except KeyError:
                pass
            try:
                del param_dict['with_DMI']
            except KeyError:
                pass
            try:
                del param_dict['magnetic_elements']
            except KeyError:
                pass
            
        cmdline_params = [
            '--fdf_fname', fdf_fname, 
            '--elements', ' '.join([str(x) for x in self.inputs.elements]),
            '--np', str(np), 
            '--output_path', '.'
        ]
        cmdline_params += ['--kmesh'] + [str(i) for i in kmesh]
        if 'parameters' in self.inputs:
            cmdline_params += [ element 
                for parameter, value in param_dict.items() for element in ('--' + parameter, str(value)) 
            ]
         
        remote_copy_list = []
        remote_copy_list.append((
            siesta_remote.computer.uuid, os.path.join(siesta_path, self._restart_copy_from), self._restart_copy_to))
        remote_copy_list.append((
            siesta_remote.computer.uuid, os.path.join(siesta_path, fdf_fname), self._restart_copy_to))

        codeinfo = CodeInfo()
        codeinfo.cmdline_params = cmdline_params
        codeinfo.code_uuid = code.uuid

        calcinfo = CalcInfo()
        calcinfo.uuid = str(self.uuid)
        calcinfo.local_copy_list = []
        calcinfo.remote_copy_list = remote_copy_list
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = ['TB2J.pickle']

        return calcinfo
