from aiida.parsers import Parser
from aiida.common import exceptions

class BasicParser(Parser):

    def parse(self, **kwargs):

        from aiida.engine import ExitCode

        try:
            output_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        try:
            filename_exchange = [element for element in output_folder.list_object_names() if '.pickle' in element][0]
        except IndexError:
            return self.exit_codes.ERROR_OUTPUT_PICKLE_MISSING

        return ExitCode(0) 
