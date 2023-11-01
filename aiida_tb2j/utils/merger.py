import pickle
from TB2J.io_merge import Merger as TB2JMerger
from TB2J.io_merge import SpinIO_merge

def read_pickle(folder_data, projv_tolerance=1e-2):

    try:
        filename = [element for element in folder_data.list_object_names() if '.pickle' in element][0]
    except IndexError:
        raise FileNotFoundError("The 'TB2J.pickle' file is not present in the FolderData object.")

    with folder_data.base.repository.open(filename, 'rb') as File:
        content = pickle.load(File)

    spin_obj = SpinIO_merge(atoms=[], spinat=[], charges=[], index_spin=[])
    spin_obj.__dict__.update(content)
    spin_obj._set_projection_vectors(projv_tolerance=projv_tolerance)

    return spin_obj

class Merger(TB2JMerger):

    def __init__(self, *folders, main_folder=None, method='structure', tolerance=1e-2):
        from copy import deepcopy
        if method not in ['structure', 'spin']:
            raise ValueError(f"Unrecognized method '{method}'. Available options are: 'structure' or 'spin'.")
        self.method = method
        self.dat = [read_pickle(folder, projv_tolerance=tolerance) for folder in folders]

        if main_folder is None:
            self.main_dat = deepcopy(self.dat[-1])
        else:
            self.main_dat = read_pickle(main_folder, projv_tolerance=tolerance)
            self.dat.append(deepcopy(self.main_dat))

        self._set_projv()
