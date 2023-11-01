import numpy as np
from copy import deepcopy
from ..data import ExchangeData

def write_unitcell_file(exchange, filename, isotropic, max_distance, mat_indeces):

    from ase.units import J

    arraynames = exchange.get_arraynames()
    if ('Jani' in arraynames or 'DMI' in arraynames) and not isotropic:
        tensor = exchange.get_exchange_tensor(with_Jani=True, with_DMI=True)
        tensor = tensor.reshape(tensor.shape[0:2] + (9,))
        mode = 'tensorial'
    else:
        tensor = deepcopy(exchange.get_Jiso())
        mode = 'isotropic'
    tensor *= 2/J

    sites = exchange.sites
    with open(filename, 'w', encoding='utf8') as File:
        cell = exchange.cell
        lattice_vectors = np.linalg.norm(cell, axis=-1)
        File.write('# Unit cell size (Angstrom):\n')
        np.savetxt(File, [lattice_vectors], fmt='%f')
        File.write('# Unit cell lattice vectors:\n')
        np.savetxt(File, cell / lattice_vectors, fmt='%f')

        pairs = np.array( exchange.pairs )
        magnetic_indices = sorted( set(np.reshape(pairs, (-1,))) )
        nspins = len(magnetic_indices)
        File.write(f'# Atoms\n{nspins} {nspins}\n')
        for i in range(nspins):
            position = np.linalg.solve( cell.T, sites[magnetic_indices[i]].position ).T % 1.0
            File.write(f'{i} {position[0]} {position[1]} {position[2]} {mat_indeces[i]}\n')

        lines = []
        formatter = {"float_kind": lambda x : "%.15e" % x}
        vectors = exchange.get_vectors().round(5)
        for i in range(len(pairs)):
            ki, li = pairs[i]
            k, l = (magnetic_indices.index(ki), magnetic_indices.index(li))
            for j in range(tensor.shape[1]):
                vec = vectors[i, j]
                J_data = tensor[i, j]
                distance = np.linalg.norm(vec @ cell)
                if (vec != np.zeros(3)).any() and distance <= max_distance:
                    vec = vec.astype(int)
                    J_string = np.array2string(J_data, formatter=formatter, max_line_width=225).strip('[]')
                    lines.append((f'{k} {l} {vec[0]} {vec[1]} {vec[2]} {J_string}\n', distance))
                    if k != l:
                        vec *= -1
                        lines.append((f'{l} {k} {vec[0]} {vec[1]} {vec[2]} {J_string}\n', distance))
        lines.sort(key=lambda x : x[-1] )
        lines = [pair[0] for pair in lines]
        File.write(f'# Interactions\n{len(lines)} {mode}\n')
        File.writelines([f'{i} ' + lines[i] for i in range(len(lines))])

def write_mat_file(exchange, filename, indeces):

    template = '''
#---------------------------------------------------
# Material {idx}
#---------------------------------------------------
material[{idx}]:material-name={name}
material[{idx}]:damping-constant=1.0
material[{idx}]:atomic-spin-moment={ms} !muB
material[{idx}]:uniaxial-anisotropy-constant=0.0
material[{idx}]:material-element={name}
material[{idx}]:initial-spin-direction = {spinat}
material[{idx}]:uniaxial-anisotropy-direction = 0.0, 0.0, 1.0
#---------------------------------------------------'''

    with open(filename, 'w', encoding='utf8') as File:
        File.write(f'material:num-materials = {len(indeces)}')
        n = 1
        for i in indeces:
            atom = exchange.sites[i]
            magmom = atom.magmom
            ms = np.linalg.norm(magmom)
            spinat = magmom if magmom.shape == (3,) else [0.0, 0.0, ms]
            text = template.format(
                idx=indeces.index(i)+1,
                name=atom.kind_name,
                ms=ms,
                spinat=','.join([str(x) for x in spinat])
            )
            File.write('\n' + text)

def write_vampire_files(
        exchange: ExchangeData, 
        material_filename: str = 'vampire.mat',
        UCF_filename: str = 'vampire.UCF',
        isotropic: bool = False,
        max_distance: float = np.inf,
        ):

    idx = sorted( set([pair[0] for pair in exchange.pairs]) )
    magmoms = exchange.magmoms().round(2)
    if len(magmoms.shape) != 2:
        magmoms = np.stack([np.zeros(magmoms.shape)]*2 + [magmoms], axis=1)
    unique_magmoms = np.array(list( set(tuple(a) for a in magmoms[idx]) ))

    representative_indeces = [
        np.where( (magmoms == vec).all(axis=-1) )[0][0] for vec in unique_magmoms
    ]
    material_indeces = [
        np.where( (unique_magmoms == vec).all(axis=-1) )[0][0] for vec in magmoms[idx]
    ]

    write_mat_file(exchange, material_filename, representative_indeces)
    write_unitcell_file(exchange, UCF_filename, isotropic, max_distance, material_indeces)
