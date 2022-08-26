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
        'supercell_size'
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
        for k, v in params.items():
            if k in floattype_keys:
                if not isinstance(v, (float, int)):
                    return f"'{k}' option must be a real number."
                if k == 'supercell_size' and v < 0.0:
                    return "'supercell_size' option must be positive."
            elif k in booltype_keys:
                if not isinstance(v, bool):
                    return f"'{k}' option must be a boolean."
            elif k in inttype_keys:
                if not isinstance(v, int):
                    return f"'{k}' option must be an integer."
            elif k in unvalid_keys:
                return f"You can't specify the '{k}' option in the parameters input port."
            elif k == 'kmesh':
                try:
                    kmesh = list(v)
                    if len(kmesh) != 3 or not all(isinstance(x, int) for x in kmesh):
                        return "The 'kmesh' option must contain 3 integers."
                except TypeError:
                    return "The 'kmesh' option must be an iterable."
            elif k == 'magnetic_elements':
                pass
            else:
                return f"Unrecognized option '{k}'"               

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
            help="TB2J siesta binary."
        )
        spec.input(
            'siesta_remote', 
            valid_type=orm.RemoteData, 
            required=True, 
            help="Parent remote folder."
        )
        spec.input(
            'elements', 
            valid_type=orm.List, 
            required=True,
            validator=orm.nodes.data.structure.validate_symbols_tuple, 
            help="List of magnetic elements."
        )
        spec.input(
            'parameters', 
            valid_type=orm.Dict,
            required=False,
            validator=validate_parameters,
            help="Contains all the optional parameters for the TB2J calculation."
        )
        spec.input(
            'structure',
            valid_type=orm.StructureData,
            required=False,
            help="Only required if the 'kmesh' variable needs to be set up based on the unit cell dimensions or some of the structure information must appear on the ExchangeData output."
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
        kmesh = [5, 5, 5]; kmesh_is_present = False;
        supercell_size = 20.0
        np = self.inputs.metadata.options.resources['num_cores_per_machine']

        if 'parameters' in self.inputs:
            param_dict = self.inputs.parameters.get_dict()
            try:
                kmesh = param_dict.pop('kmesh')
                kmesh_is_present = True
            except KeyError:
                pass
            try:
                supercell_size = param_dict.pop('supercell_size')
            except KeyError:
                pass
        if 'structure' in self.inputs:
            if kmesh_is_present:
                self.report(
                    "Both the 'kmesh' variable and a 'StructureData' object were given as inputs." 
                    "The 'kmesh' variable will be used and the StructureData input will be ignored."
                )
            else:
                from numpy import ceil
                from numpy.linalg import norm
                structure = self.inputs.structure
                lattice_parameters = norm(structure.cell, axis=-1)
                kmesh = ceil(supercell_size / lattice_parameters).astype(int).tolist()
                kmesh = [kmesh[i] if structure.pbc[i] else 1 for i in range(3)]

            
        cmdline_params = [
            '--fdf_fname', fdf_fname,
            '--np', str(np), 
            '--output_path', '.'
        ]
        cmdline_params += ['--elements'] + [str(element) for element in self.inputs.elements]
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
