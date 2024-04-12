from .. import _keys
import ase
import numpy as np
import torch
import re
import e3nn.o3 as o3
import h5py
import logging
from dptb.utils.constants import anglrMId, OPENMX2DeePTB

log = logging.getLogger(__name__)

def block_to_feature(data, idp, blocks=False, overlap_blocks=False):
    # Hamiltonian_blocks should be a h5 group in the current version
    assert blocks != False or overlap_blocks!=False, "Both feature block and overlap blocks are not provided."
    
    if blocks:
        onsite_ham = []
        edge_ham = []
    if overlap_blocks:
        edge_overlap = []

    idp.get_orbital_maps()
    idp.get_orbpair_maps()

    atomic_numbers = data[_keys.ATOMIC_NUMBERS_KEY]

    # onsite features
    if blocks:
        for atom in range(len(atomic_numbers)):
            block_index = '_'.join(map(str, map(int, [atom+1, atom+1] + list([0, 0, 0]))))
            try:
                block = blocks[block_index]
            except:
                raise IndexError("Hamiltonian block for onsite not found, check Hamiltonian file.")

            if isinstance(block, torch.Tensor):
                block = block.cpu().detach().numpy()
            symbol = ase.data.chemical_symbols[atomic_numbers[atom]]
            basis_list = idp.basis[symbol]
            onsite_out = np.zeros(idp.reduced_matrix_element)

            for index, basis_i in enumerate(basis_list):
                slice_i = idp.orbital_maps[symbol][basis_i]  
                for basis_j in basis_list[index:]:
                    slice_j = idp.orbital_maps[symbol][basis_j]
                    block_ij = block[slice_i, slice_j]
                    full_basis_i = idp.basis_to_full_basis[symbol][basis_i]
                    full_basis_j = idp.basis_to_full_basis[symbol][basis_j]

                    # fill onsite vector
                    pair_ij = full_basis_i + "-" + full_basis_j
                    feature_slice = idp.orbpair_maps[pair_ij]
                    onsite_out[feature_slice] = block_ij.flatten()

            onsite_ham.append(onsite_out)
        #onsite_ham = np.array(onsite_ham)

    # edge features
    edge_index = data[_keys.EDGE_INDEX_KEY]
    edge_cell_shift = data[_keys.EDGE_CELL_SHIFT_KEY]

    for atom_i, atom_j, R_shift in zip(edge_index[0], edge_index[1], edge_cell_shift):
        block_index = '_'.join(map(str, map(int, [atom_i+1, atom_j+1] + list(R_shift))))
        r_index = '_'.join(map(str, map(int, [atom_j+1, atom_i+1] + list(-R_shift))))
        symbol_i = ase.data.chemical_symbols[atomic_numbers[atom_i]]
        symbol_j = ase.data.chemical_symbols[atomic_numbers[atom_j]]

        # try:
        #     block = Hamiltonian_blocks[block_index]
        #     if overlap_blocks:
        #         block_s = overlap_blocks[block_index]
        # except:
        #     raise IndexError("Hamiltonian block for hopping not found, r_cut may be too big for input R.")
        if blocks:
            block = blocks.get(block_index, None)
            if block is None:
                block = blocks.get(r_index, None)
                if block is not None:
                    block = block.T
            if block is None:
                block = torch.zeros(idp.norbs[symbol_i], idp.norbs[symbol_j])
                log.warning("Hamiltonian block for hopping {} not found, r_cut may be too big for input R.".format(block_index))

                assert block.shape == (idp.norbs[symbol_i], idp.norbs[symbol_j])
            if isinstance(block, torch.Tensor):
                block = block.cpu().detach().numpy()
        if overlap_blocks:
            block_s = overlap_blocks.get(block_index, None)
            if block_s is None:
                block_s = overlap_blocks.get(r_index, None)
                if block_s is not None:
                    block_s = block_s.T
            if block_s is None:
                block_s = torch.zeros(idp.norbs[symbol_i], idp.norbs[symbol_j])
                log.warning("Overlap block for hopping {} not found, r_cut may be too big for input R.".format(block_index))

                assert block_s.shape == (idp.norbs[symbol_i], idp.norbs[symbol_j])
            
            if isinstance(block_s, torch.Tensor):
                block_s = block_s.cpu().detach().numpy()
        
        basis_i_list = idp.basis[symbol_i]
        basis_j_list = idp.basis[symbol_j]
        if blocks:
            hopping_out = np.zeros(idp.reduced_matrix_element)
        if overlap_blocks:
            overlap_out = np.zeros(idp.reduced_matrix_element)

        for basis_i in basis_i_list:
            slice_i = idp.orbital_maps[symbol_i][basis_i]
            for basis_j in basis_j_list:
                slice_j = idp.orbital_maps[symbol_j][basis_j]
                full_basis_i = idp.basis_to_full_basis[symbol_i][basis_i]
                full_basis_j = idp.basis_to_full_basis[symbol_j][basis_j]
                if idp.full_basis.index(full_basis_i) <= idp.full_basis.index(full_basis_j):
                    pair_ij = full_basis_i + "-" + full_basis_j
                    feature_slice = idp.orbpair_maps[pair_ij]
                    if blocks:
                        block_ij = block[slice_i, slice_j]
                        hopping_out[feature_slice] = block_ij.flatten()
                    if overlap_blocks:
                        block_s_ij = block_s[slice_i, slice_j]
                        overlap_out[feature_slice] = block_s_ij.flatten()

        if blocks:
            edge_ham.append(hopping_out)
        if overlap_blocks:
            edge_overlap.append(overlap_out)

    if blocks:
        data[_keys.NODE_FEATURES_KEY] = torch.as_tensor(np.array(onsite_ham), dtype=torch.get_default_dtype())
        data[_keys.EDGE_FEATURES_KEY] = torch.as_tensor(np.array(edge_ham), dtype=torch.get_default_dtype())
    if overlap_blocks:
        data[_keys.EDGE_OVERLAP_KEY] = torch.as_tensor(np.array(edge_overlap), dtype=torch.get_default_dtype())

def feature_to_block(data, idp):
    idp.get_orbital_maps()
    idp.get_orbpair_maps()

    has_block = False
    if data.get(_keys.NODE_FEATURES_KEY, None) is not None:
        node_features = data[_keys.NODE_FEATURES_KEY]
        edge_features = data[_keys.EDGE_FEATURES_KEY]
        has_block = True
        blocks = {}

    idp.get_orbital_maps()
    idp.get_orbpair_maps()

    if has_block:
        # get node blocks from node_features
        for atom, onsite in enumerate(node_features):
            symbol = ase.data.chemical_symbols[idp.untransform(data[_keys.ATOM_TYPE_KEY][atom].reshape(-1))]
            basis_list = idp.basis[symbol]
            block = torch.zeros((idp.norbs[symbol], idp.norbs[symbol]), device=node_features.device, dtype=node_features.dtype)

            for index, basis_i in enumerate(basis_list):
                slice_i = idp.orbital_maps[symbol][basis_i]
                li = anglrMId[re.findall(r"[a-zA-Z]+", basis_i)[0]]
                for basis_j in basis_list[index:]:
                    lj = anglrMId[re.findall(r"[a-zA-Z]+", basis_j)[0]]
                    slice_j = idp.orbital_maps[symbol][basis_j]
                    pair_ij = basis_i + "-" + basis_j
                    feature_slice = idp.orbpair_maps[pair_ij]
                    block_ij = onsite[feature_slice].reshape(2*li+1, 2*lj+1)
                    block[slice_i, slice_j] = block_ij
                    if slice_i != slice_j:
                        block[slice_j, slice_i] = block_ij.T

            block_index = '_'.join(map(str, map(int, [atom+1, atom+1] + list([0, 0, 0]))))
            blocks[block_index] = block
        
        # get edge blocks from edge_features
        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_cell_shift = data[_keys.EDGE_CELL_SHIFT_KEY]
        for edge, hopping in enumerate(edge_features):
            atom_i, atom_j, R_shift = edge_index[0][edge], edge_index[1][edge], edge_cell_shift[edge]
            symbol_i = ase.data.chemical_symbols[idp.untransform(data[_keys.ATOM_TYPE_KEY][atom_i].reshape(-1))]
            symbol_j = ase.data.chemical_symbols[idp.untransform(data[_keys.ATOM_TYPE_KEY][atom_j].reshape(-1))]
            block = torch.zeros((idp.norbs[symbol_i], idp.norbs[symbol_j]), device=edge_features.device, dtype=edge_features.dtype)

            for index, f_basis_i in enumerate(idp.full_basis):
                basis_i = idp.full_basis_to_basis[symbol_i].get(f_basis_i)
                if basis_i is None:
                    continue
                li = anglrMId[re.findall(r"[a-zA-Z]+", basis_i)[0]]
                slice_i = idp.orbital_maps[symbol_i][basis_i]
                for f_basis_j in idp.full_basis[index:]:
                    basis_j = idp.full_basis_to_basis[symbol_j].get(f_basis_j)
                    if basis_j is None:
                        continue
                    lj = anglrMId[re.findall(r"[a-zA-Z]+", basis_j)[0]]
                    slice_j = idp.orbital_maps[symbol_j][basis_j]
                    pair_ij = basis_i + "-" + basis_j
                    feature_slice = idp.orbpair_maps[pair_ij]
                    block_ij = hopping[feature_slice].reshape(2*li+1, 2*lj+1)
                    if f_basis_i == f_basis_j:
                        block[slice_i, slice_j] = 0.5 * block_ij
                    else:
                        block[slice_i, slice_j] = block_ij

            block_index = '_'.join(map(str, map(int, [atom_i+1, atom_j+1] + list(R_shift))))
            if atom_i < atom_j:
                if blocks.get(block_index, None) is None:
                    blocks[block_index] = block
                else:
                    blocks[block_index] += block
            elif atom_i == atom_j:
                r_index = '_'.join(map(str, map(int, [atom_i+1, atom_j+1] + list(-R_shift))))
                if blocks.get(r_index, None) is None:
                    blocks[block_index] = block
                else:
                    blocks[r_index] += block.T
            else:
                block_index = '_'.join(map(str, map(int, [atom_j+1, atom_i+1] + list(-R_shift))))
                if blocks.get(block_index, None) is None:
                    blocks[block_index] = block.T
                else:
                    blocks[block_index] += block.T
    return blocks


def openmx_to_deeptb(data, idp, openmx_hpath):
    # Hamiltonian_blocks should be a h5 group in the current version
    Us_openmx2wiki = OPENMX2DeePTB
    # init_rot_mat
    rot_blocks = {}
    for asym, orbs in idp.basis.items():
        b = [Us_openmx2wiki[re.findall(r"[A-Za-z]", orb)[0]] for orb in orbs]
        rot_blocks[asym] = torch.block_diag(*b)

    
    Hamiltonian_blocks = h5py.File(openmx_hpath, 'r')
    
    onsite_ham = []
    edge_ham = []

    idp.get_orbital_maps()
    idp.get_orbpair_maps()

    atomic_numbers = data[_keys.ATOMIC_NUMBERS_KEY]

    # onsite features
    for atom in range(len(atomic_numbers)):
        block_index = str([0, 0, 0, atom+1, atom+1])
        try:
            block = Hamiltonian_blocks[block_index][:]
        except:
            raise IndexError("Hamiltonian block for onsite not found, check Hamiltonian file.")

        
        symbol = ase.data.chemical_symbols[atomic_numbers[atom]]
        block = rot_blocks[symbol] @ block @ rot_blocks[symbol].T
        basis_list = idp.basis[symbol]
        onsite_out = np.zeros(idp.reduced_matrix_element)

        for index, basis_i in enumerate(basis_list):
            slice_i = idp.orbital_maps[symbol][basis_i]  
            for basis_j in basis_list[index:]:
                slice_j = idp.orbital_maps[symbol][basis_j]
                block_ij = block[slice_i, slice_j]
                full_basis_i = idp.basis_to_full_basis[symbol][basis_i]
                full_basis_j = idp.basis_to_full_basis[symbol][basis_j]

                # fill onsite vector
                pair_ij = full_basis_i + "-" + full_basis_j
                feature_slice = idp.orbpair_maps[pair_ij]
                onsite_out[feature_slice] = block_ij.flatten()

        onsite_ham.append(onsite_out)
        #onsite_ham = np.array(onsite_ham)

    # edge features
    edge_index = data[_keys.EDGE_INDEX_KEY]
    edge_cell_shift = data[_keys.EDGE_CELL_SHIFT_KEY]

    for atom_i, atom_j, R_shift in zip(edge_index[0], edge_index[1], edge_cell_shift):
        block_index = str(list(R_shift.int().numpy())+[int(atom_i)+1, int(atom_j)+1])

        symbol_i = ase.data.chemical_symbols[atomic_numbers[atom_i]]
        symbol_j = ase.data.chemical_symbols[atomic_numbers[atom_j]]

        block = Hamiltonian_blocks.get(block_index, 0)
        if block == 0:
            block = torch.zeros(idp.norbs[symbol_i], idp.norbs[symbol_j])
            log.warning("Hamiltonian block for hopping {} not found, r_cut may be too big for input R.".format(block_index))
        else:
            block = block[:]

        block = rot_blocks[symbol_i] @ block @ rot_blocks[symbol_j].T
        basis_i_list = idp.basis[symbol_i]
        basis_j_list = idp.basis[symbol_j]
        hopping_out = np.zeros(idp.reduced_matrix_element)

        for basis_i in basis_i_list:
            slice_i = idp.orbital_maps[symbol_i][basis_i]
            for basis_j in basis_j_list:
                slice_j = idp.orbital_maps[symbol_j][basis_j]
                block_ij = block[slice_i, slice_j]
                full_basis_i = idp.basis_to_full_basis[symbol_i][basis_i]
                full_basis_j = idp.basis_to_full_basis[symbol_j][basis_j]

                if idp.full_basis.index(full_basis_i) <= idp.full_basis.index(full_basis_j):
                    # fill hopping vector
                    pair_ij = full_basis_i + "-" + full_basis_j
                    feature_slice = idp.orbpair_maps[pair_ij]
                    hopping_out[feature_slice] = block_ij.flatten()

        edge_ham.append(hopping_out)

    data[_keys.NODE_FEATURES_KEY] = torch.as_tensor(np.array(onsite_ham), dtype=torch.get_default_dtype())
    data[_keys.EDGE_FEATURES_KEY] = torch.as_tensor(np.array(edge_ham), dtype=torch.get_default_dtype())
    Hamiltonian_blocks.close()