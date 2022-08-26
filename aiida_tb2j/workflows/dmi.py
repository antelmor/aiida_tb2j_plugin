from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import calcfunction, ToContext, WorkChain
from aiida_siesta.utils.tkdict import FDFDict
from .siesta import TB2JSiestaWorkChain
from ..data import ExchangeData

def get_rotated_structures(structure):

    from copy import deepcopy

    ase_object = structure.get_ase()
    ase_x = deepcopy(ase_object)
    ase_x.rotate(90, 'y', rotate_cell=True)
    ase_y = deepcopy(ase_object)
    ase_y.rotate(90, 'x', rotate_cell=True)

    Structure = orm.StructureData
    structure_x = Structure(ase=ase_x)
    structure_y = Structure(ase=ase_y)

    return {
        'x': structure_x,
        'y': structure_y,
        'z': structure
    }

def get_merger_object(path_x, path_y, path_z):

    from TB2J.io_merge import Merger

    merger = Merger(path_x, path_y, path_z, method='structure')
    merger.merge_Jiso()
    merger.merge_DMI()
    merger.merge_Jani()

    return merger         

@calcfunction
def get_exchange(folder_x, folder_y, folder_z):

    from ..parsers import correct_content, TB2JParser

    path_x, path_y, path_z = [folder.get_remote_path() for folder in [folder_x, folder_y, folder_z]]
    merger = get_merger_object(path_x, path_y, path_z)
    content = merger.dat.__dict__
    correct_content(content)

    exchange_data = TB2JParser.get_exchange_data(content)

    return exchange_data

class DMIWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(TB2JSiestaWorkChain, exclude=('metadata',)) 

        spec.outline(
            cls.checks,
            cls.run_tb2jsiesta,
            cls.return_results
        )

        spec.output('exchange', valid_type=ExchangeData, required=True)

        spec.exit_code(200, 'ERROR_TB2J_PLUGIN', message='At least one of the TB2J workflows failed.')
        spec.exit_code(201, 'ERROR_MERGE_TB2J', message='The merging of the exchange values failed.')

    def checks(self):

        param_dict = self.inputs.parameters.get_dict()
        translatedkey = FDFDict(param_dict)
        sortedkey = sorted(translatedkey.get_filtered_items())
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
            raise ValueError(
                "The DMI work chain requires non-collinear spin. Set the 'spin' parameter "
                "to either 'non-collinear' or 'spin-orbit'."
            )

    def run_tb2jsiesta(self):   

        inputs = self.exposed_inputs(TB2JSiestaWorkChain)

        if 'tb2j_options' in self.inputs:
            optio = self.inputs.tb2j_options.get_dict()
            del inputs['tb2j_options']
        else:
            optio = self.inputs.options.get_dict()
            optio['resources']['num_machines'] = 1
            optio['resources']['num_cores_per_machine'] = 1
        optio['parser_name'] = 'tb2j.basic'
        inputs['tb2j_options'] = orm.Dict( dict=optio )

        del inputs['structure']
        structures = get_rotated_structures(self.inputs.structure)
        for key, structure in structures.items():
            process = self.submit(TB2JSiestaWorkChain, **inputs, structure=structure)
            self.to_context(**{f'tb2j_{key}': process})
            self.report(
                f"Launched TB2JSiestaWorkChain<{process.pk}> for the {key}-rotated structure."
            )

    def return_results(self):

        from aiida.engine import ExitCode

        tb2j_processes = [self.ctx[f'tb2j_{i}'] for i in ['x', 'y', 'z']]
        if not all([process.is_finished_ok for process in tb2j_processes]):
            return self.exit_codes.ERROR_TB2J_PLUGIN

        folder_x, folder_y, folder_z = [process.outputs.remote_folder for process in tb2j_processes]
    
        exchange = get_exchange(folder_x, folder_y, folder_z)

        if 'array|Jiso' in exchange.attributes and 'array|DMI' in exchange.attributes:
            self.report(
                "The exchange constants including the DMI interaction and the " 
                "anisotropic exchange were obtained succesfully."
            )
        else:
            return self.exit_codes.ERROR_MERGE_TB2J

        self.out('exchange', exchange)

        return ExitCode(0)
