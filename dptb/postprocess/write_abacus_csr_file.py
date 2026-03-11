import os
import re
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from collections import defaultdict
import ase.data
from scipy.linalg import block_diag
from dftio.constants import ABACUS2DFTIO

# DFTIO -> ABACUS Transform Matrices (Spatial Only)
DFTIO2ABACUS = {l: M.T.astype(np.float32) for l, M in ABACUS2DFTIO.items()}

ORBITAL_MAP = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}
KEY_RE = re.compile(r'^\s*(-?\d+)[ _](-?\d+)[ _](-?\d+)[ _](-?\d+)[ _](-?\d+)\s*$')
H_FACTOR = 13.605698  # Ryd -> eV factor for Hamiltonian


def parse_basis_to_l_list(basis_str):
    """'2s2p1d' or 'spd' -> [0,0,1,1,2]."""
    if basis_str is None: return []
    s = str(basis_str).strip().lower()
    if not s: return []
    tokens = re.findall(r'(\d*)([spdfgh])', s)
    lst = []
    for num, ch in tokens:
        cnt = int(num) if num else 1
        if ch not in ORBITAL_MAP:
            raise ValueError(f"Unsupported orbital '{ch}' in '{basis_str}'")
        lst.extend([ORBITAL_MAP[ch]] * cnt)
    return lst


def find_basis_for_Z_or_symbol(basis_dict, Z):
    if Z in basis_dict: return basis_dict[Z]
    sym = ase.data.chemical_symbols[Z]
    for key in (sym, sym.capitalize(), sym.upper(), str(Z)):
        if key in basis_dict: return basis_dict[key]
    return None


def _transform_soc_spin_block(mat, spatial_left, spatial_right):
    """
    Helper function for SOC transform.
    Input 'mat' is in Spin-Block format: [[UU, UD], [DU, DD]].
    We must apply the spatial transform block-diagonally to preserve this structure.

    Returns: Transformed matrix in Spin-Block format.
    """
    # Create [[T_spatial, 0], [0, T_spatial]]
    soc_left = block_diag(spatial_left, spatial_left)
    soc_right = block_diag(spatial_right, spatial_right)

    # Apply Transform: (2N, 2N) @ (2N, 2M) @ (2M, 2M) -> (2N, 2M)
    return soc_left @ mat @ soc_right.T


def transform_2_ABACUS(mat, l_lefts, l_rights, is_soc=False):
    """
    Transform block from DFTIO ordering to ABACUS ordering.
    Supports both Non-SOC (spatial only) and SOC (spatial x spin).
    """
    if max(*(list(l_lefts) + list(l_rights))) > 5:
        raise NotImplementedError("Only support l = s..h.")

    # Construct Spatial Transform Matrices
    left_mats = [DFTIO2ABACUS[l] for l in l_lefts]
    right_mats = [DFTIO2ABACUS[l] for l in l_rights]

    spatial_left = block_diag(*left_mats) if left_mats else np.eye(0, dtype=np.float32)
    spatial_right = block_diag(*right_mats) if right_mats else np.eye(0, dtype=np.float32)

    if not is_soc:
        # Non-SOC: Standard Spatial Transform
        # Mat shape: (N_orb_i, N_orb_j)
        return spatial_left @ mat @ spatial_right.T
    else:
        # SOC Branch: Delegate to helper to handle Spin-Block format
        return _transform_soc_spin_block(mat, spatial_left, spatial_right)


def write_abacus_csr_format(matrix_dict, matrix_symbol, output_path, step=0, is_soc=False):
    """Write mapping 'Rx_Ry_Rz' -> csr_matrix into ABACUS text CSR."""
    if not matrix_dict:
        print(f"Warning: empty matrix_dict for {matrix_symbol}")
        return

    first = next(iter(matrix_dict))
    norbits = matrix_dict[first].shape[0]
    num_blocks = len(matrix_dict)

    with open(output_path, 'w') as f:
        f.write(f"STEP: {step}\n")
        f.write(f"Matrix Dimension of {matrix_symbol}(R): {norbits}\n")
        f.write(f"Matrix number of {matrix_symbol}(R): {num_blocks}\n")

        for r_key, sparse_mat in matrix_dict.items():
            r_vector_str = r_key.replace('_', ' ')
            nnz = int(sparse_mat.nnz)
            f.write(f"{r_vector_str} {nnz}\n")

            if nnz > 0:
                # 1. Write Data (Values)
                if not is_soc:
                    # Non-SOC: Simple floats
                    np.savetxt(f, sparse_mat.data.reshape(1, -1), fmt='%.8e', delimiter=' ')
                else:
                    # SOC: Complex "(re,im)" format
                    # Manual formatting for complex numbers to match ABACUS requirement
                    data_strs = [f"({val.real:.8e},{val.imag:.8e})" for val in sparse_mat.data]
                    f.write(" ".join(data_strs) + "\n")

                # 2. Write Indices (Cols)
                np.savetxt(f, sparse_mat.indices.reshape(1, -1), fmt='%d', delimiter=' ')

                # 3. Write Indptr (Rows)
                np.savetxt(f, sparse_mat.indptr.reshape(1, -1), fmt='%d', delimiter=' ')
            else:
                f.write("\n\n\n")


def write_blocks_to_abacus_csr(atomic_numbers, basis_dict, blocks_dict, matrix_symbol, output_path, step=0,
                               unfold_symmetry=True):
    """
    Entry function: Writes ABACUS CSR file with automatic SOC detection.
    Handles Spin-Block to Interleaved format conversion for SOC.
    """

    # --- 0. Auto-Detect SOC ---
    first_block = next(iter(blocks_dict.values()))

    is_complex_type = False
    if hasattr(first_block, "dtype"):
        if "torch" in str(type(first_block)):
            if first_block.is_complex(): is_complex_type = True
        elif np.iscomplexobj(first_block) or "complex" in str(first_block.dtype):
            is_complex_type = True
    elif hasattr(first_block, "toarray"):
        if np.iscomplexobj(first_block.data): is_complex_type = True

    # For H/S matrices, complex implies SOC
    is_soc = is_complex_type

    if matrix_symbol == 'DM' and not is_complex_type:
        pass

    # --- Helper: Unfold Symmetry ---
    def _ensure_hermitian_completeness(input_blocks):
        full_blocks = {}
        for k, v in input_blocks.items():
            k_str = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
            full_blocks[k_str] = v

        existing_keys = list(full_blocks.keys())
        for key in existing_keys:
            m = KEY_RE.match(key)
            if not m: continue
            i, j = int(m.group(1)), int(m.group(2))
            Rx, Ry, Rz = int(m.group(3)), int(m.group(4)), int(m.group(5))

            rev_key = f"{j}_{i}_{-Rx}_{-Ry}_{-Rz}"
            if rev_key not in full_blocks:
                block = full_blocks[key]
                # Conjugate Transpose logic
                if hasattr(block, "detach"):  # Torch
                    rev_block = block.detach().transpose(-1, -2)
                elif hasattr(block, "toarray"):  # Sparse
                    rev_block = block.transpose()
                else:  # Numpy
                    rev_block = np.swapaxes(block, -1, -2)

                # Apply Conjugate if SOC/Complex
                if is_soc:
                    if hasattr(rev_block, "conj"): rev_block = rev_block.conj()

                full_blocks[rev_key] = rev_block
        return full_blocks

    if unfold_symmetry:
        blocks_dict = _ensure_hermitian_completeness(blocks_dict)

    # --- Setup Atom & Basis Info ---
    atomic_numbers = np.asarray(atomic_numbers, dtype=int)
    factor = H_FACTOR if str(matrix_symbol).upper() == 'H' else 1.0

    element_l_lists = {}
    for Z in np.unique(atomic_numbers):
        basis_str = find_basis_for_Z_or_symbol(basis_dict, int(Z))
        if basis_str is None:
            element_l_lists[int(Z)] = [0]
        else:
            ll = parse_basis_to_l_list(basis_str)
            element_l_lists[int(Z)] = ll if ll else [0]

    # Calculate Offsets
    # site_norbits_spatial: Number of spatial orbitals per atom
    site_norbits_spatial = np.array([sum(2 * l + 1 for l in element_l_lists[int(Z)]) for Z in atomic_numbers],
                                    dtype=int)

    # ABACUS CSR Physical Dimension: 2 * Spatial for SOC
    if is_soc:
        site_norbits_physical = site_norbits_spatial * 2
    else:
        site_norbits_physical = site_norbits_spatial

    site_norbits_cumsum = np.cumsum(site_norbits_physical)
    norbits_total = int(site_norbits_cumsum[-1])

    # --- Processing Blocks ---
    r_vector_coo = defaultdict(lambda: {'data': [], 'rows': [], 'cols': []})

    for raw_key, small_block in blocks_dict.items():
        key = raw_key.decode() if isinstance(raw_key, (bytes, bytearray)) else str(raw_key)
        m = KEY_RE.match(key)
        if not m: continue

        i_site, j_site = int(m.group(1)), int(m.group(2))
        r_str = f"{int(m.group(3))}_{int(m.group(4))}_{int(m.group(5))}"

        l_lefts = element_l_lists[int(atomic_numbers[i_site])]
        l_rights = element_l_lists[int(atomic_numbers[j_site])]

        # Get Array
        if hasattr(small_block, "toarray"):
            block_arr = small_block.toarray()
        elif hasattr(small_block, "detach"):
            if small_block.is_cuda:
                block_arr = small_block.detach().cpu().numpy()
            else:
                block_arr = small_block.detach().numpy()
        else:
            block_arr = np.asarray(small_block)

        if block_arr.size == 0: continue

        # Ensure dtype correctness
        if is_soc:
            block_arr = block_arr.astype(np.complex64)
        else:
            block_arr = block_arr.astype(np.float32)

        # --- 1. Transform Basis (DFTIO -> ABACUS) ---
        # Note: transformed result is still in the same format as input (Spin-Block for SOC)
        transformed = transform_2_ABACUS(block_arr, l_lefts, l_rights, is_soc=is_soc)

        # --- 2. Handle SOC Layout Reordering (Spin-Block -> Interleaved) ---
        if is_soc:
            # Current format: Spin-Block [[UU, UD], [DU, DD]] (2Ni x 2Nj)
            # Target format: Interleaved (Atom -> Orb -> Spin) for ABACUS CSR

            spatial_ni = site_norbits_spatial[i_site]
            spatial_nj = site_norbits_spatial[j_site]

            # Revert the logic used in parse_matrix:
            # 1. Reshape to separate Spin and Orbital dimensions
            reshaped = transformed.reshape(2, spatial_ni, 2, spatial_nj)

            # 2. Transpose from (Spin_R, Orb_R, Spin_C, Orb_C)
            #               to  (Orb_R, Spin_R, Orb_C, Spin_C)
            reordered = reshaped.transpose(1, 0, 3, 2)

            # 3. Flatten back to 2D for COO generation
            transformed = reordered.reshape(2 * spatial_ni, 2 * spatial_nj)

        # --- Calculate Offsets ---
        row_offset = int(site_norbits_cumsum[i_site] - site_norbits_physical[i_site])
        col_offset = int(site_norbits_cumsum[j_site] - site_norbits_physical[j_site])

        # Convert to COO
        coo = coo_matrix(transformed)
        if coo.nnz == 0: continue

        r_vector_coo[r_str]['data'].append(coo.data / factor)
        r_vector_coo[r_str]['rows'].append((coo.row + row_offset).astype(int))
        r_vector_coo[r_str]['cols'].append((coo.col + col_offset).astype(int))

    # --- Assemble and Write ---
    reassembled = {}
    dtype_final = np.complex64 if is_soc else np.float32

    for r_str, parts in r_vector_coo.items():
        if not parts['data']:
            full = csr_matrix((norbits_total, norbits_total), dtype=dtype_final)
        else:
            data = np.concatenate(parts['data']).astype(dtype_final)
            rows = np.concatenate(parts['rows']).astype(int)
            cols = np.concatenate(parts['cols']).astype(int)
            full = csr_matrix((data, (rows, cols)), shape=(norbits_total, norbits_total))
        reassembled[r_str] = full

    write_abacus_csr_format(reassembled, matrix_symbol, output_path, step=step, is_soc=is_soc)
    return reassembled, norbits_total