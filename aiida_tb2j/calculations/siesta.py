from aiida import orm
from aiida_siesta.calculations.siesta import SiestaCalculation as ParentSiestaCalculation
from aiida_siesta.calculations.siesta import validate_inputs as validate_parent_inputs
from paramiko.ssh_exception import SSHException
from ..utils import read_DM

def validate_angles(value, _):

    if value:
        angles_list = value.get_list()
        if len(angles_list) != 3:
            return "You must provide a value for each cartesian axis; the angles list must excatly 3 values."
        if not all([isinstance(a, int) or isinstance(a, float) for a in angles_list]):
            return "The angles must be numeric data."

def validate_inputs(value, _):

    message = validate_parent_inputs(value, _)
    if message:
        return message
    if 'spin_angles' in value:
        if 'parent_calc_folder' not in value:
            return "If 'spin_angles' are specified, a 'parent_calc_folder' must be provided."

def get_density_matrix(remote):

    from time import sleep

    for i in range(10):
        try:
            DM = read_DM(remote)
        except SSHException:
            sleep(20)
        else:
            break

    try:
        return DM
    except NameError:
        pass

class SiestaCalculation(ParentSiestaCalculation):

    @classmethod
    def define(cls, spec):

        super().define(spec)

        spec.input(
            'spin_angles',
            valid_type=orm.List,
            validator=validate_angles,
            required=False,
            help='If provided, the spin angles of the .DM file will be rotated.'
        )

        spec.inputs.validator = validate_inputs

    def prepare_for_submission(self, folder):

        calcinfo = super().prepare_for_submission(folder)

        if 'spin_angles' in self.inputs:
            DM_file = folder.get_abs_path('aiida.DM')
            spin_angles = self.inputs.spin_angles.get_list()
            calcinfo.remote_copy_list = []
            remote = self.inputs.parent_calc_folder
            DM = get_density_matrix(remote)
            if DM is None:
                self.report(
                    "Pausing CalcJob after it failed to connect to remote folder."
                )
                self.pause()
            DMrot = DM.spin_rotate(spin_angles)
            DMrot.write(DM_file)
            calcinfo.provenance_exclude_list = ['aiida.DM']

        return calcinfo
