from .. import _keys
import ase
import numpy as np
import torch

def ham_block_to_feature(data, idp, Hamiltonian_blocks, overlap_blocks=False): 
    # Hamiltonian_blocks should be a h5 group in the current version
    onsite_ham = []
    edge_ham = []
    if overlap_blocks:
        edge_overlap = []

    idp.get_orbital_maps()
    idp.get_node_maps()
    idp.get_pair_maps()

    atomic_numbers = data[_keys.ATOMIC_NUMBERS_KEY]

    # onsite features
    for atom in range(len(atomic_numbers)):
        block_index = '_'.join(map(str, map(int, [atom+1, atom+1] + list([0, 0, 0]))))
        try:
            block = Hamiltonian_blocks[block_index]
        except:
            raise IndexError("Hamiltonian block for onsite not found, check Hamiltonian file.")

        symbol = ase.data.chemical_symbols[atomic_numbers[atom]]
        basis_list = idp.basis[symbol]
        onsite_out = np.zeros(idp.node_reduced_matrix_element)

        for index, basis_i in enumerate(basis_list):
            slice_i = idp.orbital_maps[symbol][basis_i]  
            for basis_j in basis_list[index:]:
                slice_j = idp.orbital_maps[symbol][basis_j]
                block_ij = block[slice_i, slice_j]
                full_basis_i = idp.basis_to_full_basis[symbol][basis_i]
                full_basis_j = idp.basis_to_full_basis[symbol][basis_j]

                # fill onsite vector
                pair_ij = full_basis_i + "-" + full_basis_j
                feature_slice = idp.node_maps[pair_ij]
                onsite_out[feature_slice] = block_ij.flatten()

        onsite_ham.append(onsite_out)
        #onsite_ham = np.array(onsite_ham)

    # edge features
    edge_index = data[_keys.EDGE_INDEX_KEY]
    edge_cell_shift = data[_keys.EDGE_CELL_SHIFT_KEY]

    for atom_i, atom_j, R_shift in zip(edge_index[0], edge_index[1], edge_cell_shift):
        block_index = '_'.join(map(str, map(int, [atom_i+1, atom_j+1] + list(R_shift))))
        try:
            block = Hamiltonian_blocks[block_index]
            if overlap_blocks:
                block_s = overlap_blocks[block_index]
        except:
            raise IndexError("Hamiltonian block for hopping not found, r_cut may be too big for input R.")

        symbol_i = ase.data.chemical_symbols[atomic_numbers[atom_i]]
        symbol_j = ase.data.chemical_symbols[atomic_numbers[atom_j]]
        basis_i_list = idp.basis[symbol_i]
        basis_j_list = idp.basis[symbol_j]
        hopping_out = np.zeros(idp.edge_reduced_matrix_element)
        if overlap_blocks:
            overlap_out = np.zeros(idp.edge_reduced_matrix_element)

        for basis_i in basis_i_list:
            slice_i = idp.orbital_maps[symbol_i][basis_i]
            for basis_j in basis_j_list:
                slice_j = idp.orbital_maps[symbol_j][basis_j]
                block_ij = block[slice_i, slice_j]
                if overlap_blocks:
                    block_s_ij = block_s[slice_i, slice_j]
                full_basis_i = idp.basis_to_full_basis[symbol_i][basis_i]
                full_basis_j = idp.basis_to_full_basis[symbol_j][basis_j]

                # fill hopping vector
                pair_ij = full_basis_i + "-" + full_basis_j
                feature_slice = idp.pair_maps[pair_ij]
                hopping_out[feature_slice] = block_ij.flatten()
                if overlap_blocks:
                    overlap_out[feature_slice] = block_s_ij.flatten()

        edge_ham.append(hopping_out)
        if overlap_blocks:
            edge_overlap.append(overlap_out)

    data[_keys.NODE_FEATURES_KEY] = torch.as_tensor(np.array(onsite_ham), dtype=torch.get_default_dtype())
    data[_keys.EDGE_FEATURES_KEY] = torch.as_tensor(np.array(edge_ham), dtype=torch.get_default_dtype())
    if overlap_blocks:
        data[_keys.OVERLAP_KEY] = torch.as_tensor(np.array(edge_overlap), dtype=torch.get_default_dtype())