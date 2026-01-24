from .. import _keys
import ase
import numpy as np
import torch
import re
import e3nn.o3 as o3
import h5py
import logging
from dptb.utils.constants import OPENMX2DeePTB
from dptb.data import AtomicData, AtomicDataDict

log = logging.getLogger(__name__)

def block_to_feature(data, idp, blocks=False, overlap_blocks=False, orthogonal=False):
    # Hamiltonian_blocks should be a h5 group in the current version
    assert blocks != False or overlap_blocks!=False, "Both feature block and overlap blocks are not provided."
    if blocks != False:
        proto_block = blocks[list(blocks.keys())[0]][:]
    else:
        proto_block = overlap_blocks[list(overlap_blocks.keys())[0]][:]

    if blocks:
        onsite_ham = []
    if overlap_blocks and not orthogonal:
        onsite_ovp = []

    idp.get_orbital_maps()
    idp.get_orbpair_maps()

    dtype = proto_block.dtype
    
    if isinstance(proto_block, torch.Tensor):
        meta_dtype = torch
    elif isinstance(proto_block, np.ndarray):
        meta_dtype = np
    else:
        raise TypeError("Hamiltonian blocks should be either torch.Tensor or np.ndarray.")

    if isinstance(data, AtomicData):
        if not hasattr(data, _keys.ATOMIC_NUMBERS_KEY):
            setattr(data, _keys.ATOMIC_NUMBERS_KEY, idp.untransform(data[_keys.ATOM_TYPE_KEY]))
    if isinstance(data, dict):
        if not data.get(_keys.ATOMIC_NUMBERS_KEY, None):
            data[_keys.ATOMIC_NUMBERS_KEY] = idp.untransform(data[_keys.ATOM_TYPE_KEY])
    atomic_numbers = data[_keys.ATOMIC_NUMBERS_KEY]

    # onsite features
    if blocks or overlap_blocks:
        if blocks:
            if blocks.get("0_0_0_0_0") is None:
                start_id = 1
            else:
                start_id = 0
        else:
            if overlap_blocks.get("0_0_0_0_0") is None:
                start_id = 1
            else:
                start_id = 0


        for atom in range(len(atomic_numbers)):
            block_index = '_'.join(map(str, map(int, [atom+start_id, atom+start_id] + list([0, 0, 0]))))

            if blocks:
                try:
                    block = blocks[block_index][:]
                except:
                    raise IndexError("Hamiltonian block for onsite not found, check Hamiltonian file.")
            
            if overlap_blocks and not orthogonal:
                try:
                    overlap_block = overlap_blocks[block_index][:]
                except:
                    raise IndexError("Overlap block for onsite not found, check Overlap file.")
                
                onsite_ovp_out = meta_dtype.zeros(idp.reduced_matrix_element)

            # if isinstance(block, torch.Tensor):
            #     block = block.cpu().detach().numpy()
            symbol = ase.data.chemical_symbols[atomic_numbers[atom]]
            basis_list = idp.basis[symbol]
            onsite_out = meta_dtype.zeros(idp.reduced_matrix_element)

            for index, basis_i in enumerate(basis_list):
                slice_i = idp.orbital_maps[symbol][basis_i]  
                for basis_j in basis_list[index:]:
                    slice_j = idp.orbital_maps[symbol][basis_j]
                    full_basis_i = idp.basis_to_full_basis[symbol][basis_i]
                    full_basis_j = idp.basis_to_full_basis[symbol][basis_j]
                    pair_ij = full_basis_i + "-" + full_basis_j
                    feature_slice = idp.orbpair_maps[pair_ij]

                    if blocks:
                        block_ij = block[slice_i, slice_j]
                        onsite_out[feature_slice] = block_ij.flatten()

                    if overlap_blocks and not orthogonal:
                        overlap_block_ij = overlap_block[slice_i, slice_j]
                        onsite_ovp_out[feature_slice] = overlap_block_ij.flatten()

            if blocks:
                onsite_ham.append(onsite_out)
            if overlap_blocks and not orthogonal:
                onsite_ovp.append(onsite_ovp_out)
        #onsite_ham = np.array(onsite_ham)

    # edge features
    edge_index = data[_keys.EDGE_INDEX_KEY]
    edge_cell_shift = data[_keys.EDGE_CELL_SHIFT_KEY]
    edge_type = idp.transform_bond(*data[_keys.ATOMIC_NUMBERS_KEY][edge_index]).flatten()
    if overlap_blocks:
        ovp_features = torch.zeros(len(edge_index[0]), idp.reduced_matrix_element)
    if blocks:
        edge_features = torch.zeros(len(edge_index[0]), idp.reduced_matrix_element)

    if blocks or overlap_blocks:
        for bt in range(len(idp.bond_types)):

            symbol_i, symbol_j = idp.bond_types[bt].split("-")
            basis_i = idp.basis[symbol_i]
            basis_j = idp.basis[symbol_j]
            mask = edge_type.eq(bt)
            if torch.all(~mask):
                continue
            b_edge_index = edge_index[:, mask]
            b_edge_cell_shift = edge_cell_shift[mask]
            ijR = torch.cat([b_edge_index.T+start_id, b_edge_cell_shift], dim=1).int().tolist()
            rev_ijR = torch.cat([b_edge_index[[1, 0]].T+start_id, -b_edge_cell_shift], dim=1).int().tolist()
            ijR = list(map(lambda x: '_'.join(map(str, x)), ijR))
            rev_ijR = list(map(lambda x: '_'.join(map(str, x)), rev_ijR))

            if blocks:
                b_blocks = []
                for i,j in zip(ijR, rev_ijR):
                    if i in blocks:
                        b_blocks.append(blocks[i][:])
                    elif j in blocks:
                        b_blocks.append(blocks[j][:].T)
                    else:
                        b_blocks.append(meta_dtype.zeros((idp.norbs[symbol_i], idp.norbs[symbol_j]), dtype=dtype))
                # b_blocks = [blocks[i] if i in blocks else blocks[j].T for i,j in zip(ijR, rev_ijR)]
                if isinstance(b_blocks[0], torch.Tensor):
                    b_blocks = torch.stack(b_blocks, dim=0)
                else:
                    b_blocks = np.stack(b_blocks, axis=0)

            if overlap_blocks:
                s_blocks = []
                for i,j in zip(ijR, rev_ijR):
                    if i in overlap_blocks:
                        s_blocks.append(overlap_blocks[i][:])
                    elif j in overlap_blocks:
                        s_blocks.append(overlap_blocks[j][:].T)
                    else:
                        s_blocks.append(meta_dtype.zeros((idp.norbs[symbol_i], idp.norbs[symbol_j]), dtype=dtype))
                # s_blocks = [overlap_blocks[i] if i in overlap_blocks else overlap_blocks[j].T for i,j in zip(ijR, rev_ijR)]
                if isinstance(s_blocks[0], torch.Tensor):
                    s_blocks = torch.stack(s_blocks, dim=0)
                else:
                    s_blocks = np.stack(s_blocks, axis=0)

            for orb_i in basis_i:
                slice_i = idp.orbital_maps[symbol_i][orb_i]
                for orb_j in basis_j:
                    slice_j = idp.orbital_maps[symbol_j][orb_j]
                    full_orb_i = idp.basis_to_full_basis[symbol_i][orb_i]
                    full_orb_j = idp.basis_to_full_basis[symbol_j][orb_j]
                    if idp.full_basis.index(full_orb_i) <= idp.full_basis.index(full_orb_j):
                        pair_ij = full_orb_i + "-" + full_orb_j
                        feature_slice = idp.orbpair_maps[pair_ij]
                        if blocks:
                            edge_features[mask, feature_slice] = torch.as_tensor(b_blocks[:,slice_i, slice_j].reshape(b_edge_index.shape[1], -1), dtype=torch.get_default_dtype())
                        if overlap_blocks:
                            ovp_features[mask, feature_slice] = torch.as_tensor(s_blocks[:,slice_i, slice_j].reshape(b_edge_index.shape[1], -1), dtype=torch.get_default_dtype())

    if blocks:
        if isinstance(onsite_ham[0], torch.Tensor):
            onsite_ham = torch.stack(onsite_ham, dim=0)
        else:
            onsite_ham = np.stack(onsite_ham, axis=0)
        data[_keys.NODE_FEATURES_KEY] = torch.as_tensor(onsite_ham, dtype=torch.get_default_dtype())
        data[_keys.EDGE_FEATURES_KEY] = edge_features
    if overlap_blocks:
        if not orthogonal:
            if isinstance(onsite_ovp[0], torch.Tensor):
                onsite_ovp = torch.stack(onsite_ovp, dim=0)
            else:
                onsite_ovp = np.stack(onsite_ovp, axis=0)
            data[_keys.NODE_OVERLAP_KEY] = torch.as_tensor(onsite_ovp, dtype=torch.get_default_dtype())
        data[_keys.EDGE_OVERLAP_KEY] = ovp_features

# def block_to_feature(data, idp, blocks=False, overlap_blocks=False):
#     # Hamiltonian_blocks should be a h5 group in the current version
#     assert blocks != False or overlap_blocks!=False, "Both feature block and overlap blocks are not provided."
    
#     if blocks:
#         onsite_ham = []
#         edge_ham = []
#     if overlap_blocks:
#         edge_overlap = []

#     idp.get_orbital_maps()
#     idp.get_orbpair_maps()

#     if isinstance(data, AtomicData):
#         if not hasattr(data, _keys.ATOMIC_NUMBERS_KEY):
#             setattr(data, _keys.ATOMIC_NUMBERS_KEY, idp.untransform(data[_keys.ATOM_TYPE_KEY]))
#     if isinstance(data, dict):
#         if not data.get(_keys.ATOMIC_NUMBERS_KEY, None):
#             data[_keys.ATOMIC_NUMBERS_KEY] = idp.untransform(data[_keys.ATOM_TYPE_KEY])
#     atomic_numbers = data[_keys.ATOMIC_NUMBERS_KEY]

#     # onsite features
#     if blocks:
#         for atom in range(len(atomic_numbers)):
#             block_index = '_'.join(map(str, map(int, [atom+1, atom+1] + list([0, 0, 0]))))
#             try:
#                 block = blocks[block_index]
#             except:
#                 raise IndexError("Hamiltonian block for onsite not found, check Hamiltonian file.")

#             if isinstance(block, torch.Tensor):
#                 block = block.cpu().detach().numpy()
#             symbol = ase.data.chemical_symbols[atomic_numbers[atom]]
#             basis_list = idp.basis[symbol]
#             onsite_out = np.zeros(idp.reduced_matrix_element)

#             for index, basis_i in enumerate(basis_list):
#                 slice_i = idp.orbital_maps[symbol][basis_i]  
#                 for basis_j in basis_list[index:]:
#                     slice_j = idp.orbital_maps[symbol][basis_j]
#                     block_ij = block[slice_i, slice_j]
#                     full_basis_i = idp.basis_to_full_basis[symbol][basis_i]
#                     full_basis_j = idp.basis_to_full_basis[symbol][basis_j]

#                     # fill onsite vector
#                     pair_ij = full_basis_i + "-" + full_basis_j
#                     feature_slice = idp.orbpair_maps[pair_ij]
#                     onsite_out[feature_slice] = block_ij.flatten()

#             onsite_ham.append(onsite_out)
#         #onsite_ham = np.array(onsite_ham)

#     # edge features
#     edge_index = data[_keys.EDGE_INDEX_KEY]
#     edge_cell_shift = data[_keys.EDGE_CELL_SHIFT_KEY]

#     for atom_i, atom_j, R_shift in zip(edge_index[0], edge_index[1], edge_cell_shift):
#         block_index = '_'.join(map(str, map(int, [atom_i+1, atom_j+1] + list(R_shift))))
#         r_index = '_'.join(map(str, map(int, [atom_j+1, atom_i+1] + list(-R_shift))))
#         symbol_i = ase.data.chemical_symbols[atomic_numbers[atom_i]]
#         symbol_j = ase.data.chemical_symbols[atomic_numbers[atom_j]]

#         # try:
#         #     block = Hamiltonian_blocks[block_index]
#         #     if overlap_blocks:
#         #         block_s = overlap_blocks[block_index]
#         # except:
#         #     raise IndexError("Hamiltonian block for hopping not found, r_cut may be too big for input R.")
#         if blocks:
#             block = blocks.get(block_index, None)
#             if block is None:
#                 block = blocks.get(r_index, None)
#                 if block is not None:
#                     block = block.T
#             if block is None:
#                 block = torch.zeros(idp.norbs[symbol_i], idp.norbs[symbol_j])
#                 log.warning("Hamiltonian block for hopping {} not found, r_cut may be too big for input R.".format(block_index))

#                 assert block.shape == (idp.norbs[symbol_i], idp.norbs[symbol_j])
#             if isinstance(block, torch.Tensor):
#                 block = block.cpu().detach().numpy()
#         if overlap_blocks:
#             block_s = overlap_blocks.get(block_index, None)
#             if block_s is None:
#                 block_s = overlap_blocks.get(r_index, None)
#                 if block_s is not None:
#                     block_s = block_s.T
#             if block_s is None:
#                 block_s = torch.zeros(idp.norbs[symbol_i], idp.norbs[symbol_j])
#                 log.warning("Overlap block for hopping {} not found, r_cut may be too big for input R.".format(block_index))

#                 assert block_s.shape == (idp.norbs[symbol_i], idp.norbs[symbol_j])
            
#             if isinstance(block_s, torch.Tensor):
#                 block_s = block_s.cpu().detach().numpy()
        
#         basis_i_list = idp.basis[symbol_i]
#         basis_j_list = idp.basis[symbol_j]
#         if blocks:
#             hopping_out = np.zeros(idp.reduced_matrix_element)
#         if overlap_blocks:
#             overlap_out = np.zeros(idp.reduced_matrix_element)

#         for basis_i in basis_i_list:
#             slice_i = idp.orbital_maps[symbol_i][basis_i]
#             for basis_j in basis_j_list:
#                 slice_j = idp.orbital_maps[symbol_j][basis_j]
#                 full_basis_i = idp.basis_to_full_basis[symbol_i][basis_i]
#                 full_basis_j = idp.basis_to_full_basis[symbol_j][basis_j]
#                 if idp.full_basis.index(full_basis_i) <= idp.full_basis.index(full_basis_j):
#                     pair_ij = full_basis_i + "-" + full_basis_j
#                     feature_slice = idp.orbpair_maps[pair_ij]
#                     if blocks:
#                         block_ij = block[slice_i, slice_j]
#                         hopping_out[feature_slice] = block_ij.flatten()
#                     if overlap_blocks:
#                         block_s_ij = block_s[slice_i, slice_j]
#                         overlap_out[feature_slice] = block_s_ij.flatten()

#         if blocks:
#             edge_ham.append(hopping_out)
#         if overlap_blocks:
#             edge_overlap.append(overlap_out)

#     if blocks:
#         data[_keys.NODE_FEATURES_KEY] = torch.as_tensor(np.array(onsite_ham), dtype=torch.get_default_dtype())
#         data[_keys.EDGE_FEATURES_KEY] = torch.as_tensor(np.array(edge_ham), dtype=torch.get_default_dtype())
#     if overlap_blocks:
#         data[_keys.EDGE_OVERLAP_KEY] = torch.as_tensor(np.array(edge_overlap), dtype=torch.get_default_dtype())

def feature_to_block(data, idp, overlap: bool = False):
    """Convert feature vectors to Hamiltonian/overlap block matrices.

    This function uses pre-computed scatter indices for vectorized operations,
    avoiding nested Python loops for better performance.

    Args:
        data: AtomicData dictionary containing node/edge features
        idp: OrbitalMapper instance with basis information
        overlap: If True, process overlap features instead of Hamiltonian

    Returns:
        Dictionary mapping block indices to block matrices
    """
    # Get pre-computed index mappings (computed once, cached)
    node_indices = idp.get_node_feature_to_block_indices()
    edge_indices = idp.get_edge_feature_to_block_indices()

    has_block = False
    if not overlap:
        if data.get(_keys.NODE_FEATURES_KEY, None) is not None:
            node_features = data[_keys.NODE_FEATURES_KEY]
            edge_features = data[_keys.EDGE_FEATURES_KEY]
            has_block = True
            blocks = {}
    else:
        if data.get(_keys.NODE_OVERLAP_KEY, None) is not None:
            node_features = data[_keys.NODE_OVERLAP_KEY]
            edge_features = data[_keys.EDGE_OVERLAP_KEY]
            has_block = True
            blocks = {}
        else:
            raise KeyError("Overlap features not found in data.")

    if has_block:
        device = node_features.device
        dtype = node_features.dtype

        # Pre-compute atom symbols for all atoms (vectorized lookup)
        atom_types = data[_keys.ATOM_TYPE_KEY]
        atom_symbols = [
            ase.data.chemical_symbols[idp.untransform(atom_types[i].reshape(-1))]
            for i in range(len(atom_types))
        ]

        # Cache device-transferred indices per symbol to avoid redundant transfers
        unique_symbols = set(atom_symbols)
        node_indices_on_device = {}
        for symbol in unique_symbols:
            idx_info = node_indices[symbol]
            node_indices_on_device[symbol] = {
                'src': idx_info['src'].to(device),
                'dst': idx_info['dst'].to(device),
                'dst_T': idx_info['dst_T'].to(device),
                'is_diag': idx_info['is_diag'].to(device),
                'norb': idx_info['norb']
            }

        # Process node blocks using scatter
        for atom, onsite in enumerate(node_features):
            symbol = atom_symbols[atom]
            idx_info = node_indices_on_device[symbol]

            # Use cached indices (already on device)
            src_idx = idx_info['src']
            dst_idx = idx_info['dst']
            dst_idx_T = idx_info['dst_T']
            is_diag = idx_info['is_diag']
            norb = idx_info['norb']

            # Create flat block and scatter values
            block_flat = torch.zeros(norb * norb, device=device, dtype=dtype)

            # Scatter upper triangle values
            block_flat.scatter_(0, dst_idx, onsite[src_idx])

            # Scatter lower triangle (transpose) for off-diagonal blocks
            off_diag_mask = ~is_diag
            if off_diag_mask.any():
                block_flat.scatter_(0, dst_idx_T[off_diag_mask], onsite[src_idx[off_diag_mask]])

            block = block_flat.reshape(norb, norb)
            block_index = '_'.join(map(str, map(int, [atom, atom, 0, 0, 0])))
            blocks[block_index] = block

        # Process edge blocks using scatter
        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_cell_shift = data[_keys.EDGE_CELL_SHIFT_KEY]

        # Cache device-transferred indices for all possible bond types
        edge_indices_on_device = {}
        for sym_i in unique_symbols:
            for sym_j in unique_symbols:
                bond_type = f"{sym_i}-{sym_j}"
                if bond_type in edge_indices:
                    idx_info = edge_indices[bond_type]
                    edge_indices_on_device[bond_type] = {
                        'src': idx_info['src'].to(device),
                        'dst': idx_info['dst'].to(device),
                        'scale': idx_info['scale'].to(device=device, dtype=dtype),
                        'norb_i': idx_info['norb_i'],
                        'norb_j': idx_info['norb_j']
                    }

        for edge, hopping in enumerate(edge_features):
            atom_i, atom_j = int(edge_index[0][edge]), int(edge_index[1][edge])
            R_shift = edge_cell_shift[edge]
            symbol_i = atom_symbols[atom_i]
            symbol_j = atom_symbols[atom_j]
            bond_type = f"{symbol_i}-{symbol_j}"

            if bond_type not in edge_indices_on_device:  
                available = ", ".join(sorted(map(str, edge_indices.keys())))  
                msg = (  
                    f"Missing precomputed edge indices for bond type '{bond_type}'. "  
                    f"Encountered edge between atoms {atom_i} ({symbol_i}) and "  
                    f"{atom_j} ({symbol_j}) with cell shift {list(map(int, R_shift))}. "  
                    f"Available bond types in edge_indices: {available if available else 'none'}."  
                )  
                log.error(msg)  
                raise KeyError(msg) 
            idx_info = edge_indices_on_device[bond_type]

            # Use cached indices (already on device)
            src_idx = idx_info['src']
            dst_idx = idx_info['dst']
            scale = idx_info['scale']
            norb_i = idx_info['norb_i']
            norb_j = idx_info['norb_j']

            # Create flat block and scatter scaled values
            block_flat = torch.zeros(norb_i * norb_j, device=device, dtype=dtype)
            block_flat.scatter_(0, dst_idx, hopping[src_idx] * scale)
            block = block_flat.reshape(norb_i, norb_j)

            # Handle block accumulation based on atom ordering
            block_index = '_'.join(map(str, map(int, [atom_i, atom_j] + list(R_shift))))
            if atom_i < atom_j:
                if blocks.get(block_index, None) is None:
                    blocks[block_index] = block
                else:
                    blocks[block_index] += block
            elif atom_i == atom_j:
                r_index = '_'.join(map(str, map(int, [atom_i, atom_j] + list(-R_shift))))
                if blocks.get(r_index, None) is None:
                    blocks[block_index] = block
                else:
                    blocks[r_index] += block.T
            else:
                block_index = '_'.join(map(str, map(int, [atom_j, atom_i] + list(-R_shift))))
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