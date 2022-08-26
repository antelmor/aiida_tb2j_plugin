from aiida.parsers import Parser
from aiida.common import exceptions

def correct_content(content):

    from numpy import zeros

    n = max(content['index_spin']) + 1

    if content['colinear']:
        content['exchange_Jdict'].update({((0, 0, 0), i, i): 0.0 for i in range(n)})
    else: 
        shape = {
            'exchange_Jdict': (),
            'Jani_dict': (3, 3),
            'dmi_ddict': (3),
            'biquadratic_Jdict': (2)
        }
        for data_type in shape:
            content[data_type].update({((0, 0, 0), i, i): zeros(shape[data_type]) for i in range(n)})

def branched_keys(tb2j_keys, npairs):

    from numpy.linalg import norm

    msites = int( (2*npairs)**0.5 )
    branch_size = int( len(tb2j_keys)/msites**2 )
    new_keys = sorted(tb2j_keys, key=lambda x : -x[1] + x[2])[(npairs-msites)*branch_size:]
    new_keys.sort(key=lambda x : x[1:])
    bkeys = [new_keys[i:i+branch_size] for i in range(0, len(new_keys), branch_size)]

    return [sorted(branch, key=lambda x : norm(x[0])) for branch in bkeys]

class TB2JParser(Parser):

    def parse(self, **kwargs):

        from aiida.engine import ExitCode

        try:
            output_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        try:
            pickle_filename = [element for element in output_folder.list_object_names() if '.pickle' in element][0]
        except IndexError:
            return self.exit_codes.ERROR_OUTPUT_PICKLE_MISSING

        try:
            content = self._get_pickle_content(pickle_filename)
            correct_content(content)
        except (IOError, OSError):
            return self.exit_codes.ERROR_OUTPUT_PICKLE_READ

        try:
            exchange_data = self.get_exchange_data(content)
        except (IOError, OSError):
            return self.exit_codes.ERROR_OUTPUT_EXCHANGE_DATA

        self.out('exchange', exchange_data)

        parser_info = {}
        parser_info['parser_info'] = 'AiiDA TB2J Parser'
        parser_info['parser_warnings'] = []
        parser_info['output_data_filename'] = pickle_filename

        return ExitCode(0)
    
    def _get_pickle_content(self, pickle_filename):

        import pickle

        with self.retrieved.base.repository.open(pickle_filename, 'rb') as File:
            content = pickle.load(File)

        return content

    @staticmethod
    def get_exchange_data(content):

        from aiida.orm import StructureData
        from ..data import ExchangeData

        structure = StructureData(ase=content['atoms'])
        exchange = ExchangeData()
        if content['colinear']:
            exchange.non_collinear = False
            exchange.set_structure_info(structure=structure, magmoms=content['magmoms'])
        else:
            exchange.non_collinear = True
            exchange.set_structure_info(structure=structure, magmoms=content['spinat'])
        exchange.magnetic_elements = [exchange.sites[i].kind_name for i in range(len(exchange.sites)) if content['index_spin'][i] >= 0]
        
        bkeys = branched_keys(content['distance_dict'].keys(), len(exchange.pairs))
        vectors = [ [content['distance_dict'][key][0] for key in branch] for branch in bkeys ]
        exchange.set_vectors(vectors, cartesian=True)

        Jiso = [ [content['exchange_Jdict'][key] for key in branch] for branch in bkeys ]
        exchange.set_exchange_array('Jiso', Jiso)

        if not content['colinear']:
           Jani = [ [content['Jani_dict'][key] for key in branch] for branch in bkeys ]
           exchange.set_exchange_array('Jani', Jani)
           DMI = [ [content['dmi_ddict'][key] for key in branch] for branch in bkeys ]
           exchange.set_exchange_array('DMI', DMI)
           Biquad = [ [content['biquadratic_Jdict'][key] for key in branch] for branch in bkeys ]
           exchange.set_exchange_array('Biquad', Biquad)

        return exchange
