from aiida.orm import List

def get_valence_configuration(pseudo):

    content = pseudo.get_content().split('\n')
    configuration = [content[i] for i in range(*[content.index(line) for line in content if 'valence-configuration' in line])]

    return configuration

def get_magnetic_elements(pseudos):

    configuration = {pseudo: get_valence_configuration(pseudos[pseudo]) for pseudo in pseudos}

    result = []
    for pseudo in configuration:
        if [line for line in configuration[pseudo] if 'l="d"' in line and 'occupation="10"' not in line]:
            result.append(pseudo)

    if not result:
        for pseudo in configuration:
            if [line for line in configuration[pseudo] if ('l="p"' in line and 'occupation="6"' not in line) or 'l="d"' in line]:
                result.append(pseudo)        

    return List(list=result)

