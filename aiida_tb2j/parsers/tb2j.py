from aiida.parsers import Parser
from aiida.common import exceptions
from ..data.exchange import ExchangeData, correct_content

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

        pbc = None
        if 'structure' in self.node.inputs:
            pbc = self.node.inputs['structure'].pbc
        try:
            exchange_data = ExchangeData.load_tb2j(content=content, pbc=pbc)
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
