import os
import lmdb
import pickle
import re
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from collections import defaultdict
import ase.data
from scipy.linalg import block_diag
from dftio.constants import ABACUS2DFTIO

# DFTIO -> ABACUS
DFTIO2ABACUS = {l: M.T.astype(np.float32) for l, M in ABACUS2DFTIO.items()}

ORBITAL_MAP = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}
KEY_RE = re.compile(r'^\s*(-?\d+)[ _](-?\d+)[ _](-?\d+)[ _](-?\d+)[ _](-?\d+)\s*$')
H_FACTOR = 13.605698  # Ryd -> eV factor for Hamiltonian


def parse_basis_to_l_list(basis_str):
    """'2s2p1d' or 'spd' -> [0,0,1,1,2]."""
    if basis_str is None:
        return []
    s = str(basis_str).strip().lower()
    if s == "":
        return []
    tokens = re.findall(r'(\d*)([spdfgh])', s)
    lst = []
    for num, ch in tokens:
        cnt = int(num) if num else 1
        if ch not in ORBITAL_MAP:
            raise ValueError(f"Unsupported orbital '{ch}' in '{basis_str}'")
        lst.extend([ORBITAL_MAP[ch]] * cnt)
    return lst


def find_basis_for_Z_or_symbol(basis_dict, Z):
    """Find basis string for atomic number Z (multiple key forms)."""
    if Z in basis_dict:
        return basis_dict[Z]
    sym = ase.data.chemical_symbols[Z]
    for key_try in (sym, sym.capitalize(), sym.upper(), str(Z)):
        if key_try in basis_dict:
            return basis_dict[key_try]
    for k, v in basis_dict.items():
        if isinstance(k, str) and k.lower() == sym.lower():
            return v
    return None


def transform_2_ABACUS(mat, l_lefts, l_rights):
    """Transform block from DFTIO ordering to ABACUS ordering."""
    if max(*(list(l_lefts) + list(l_rights))) > 5:
        raise NotImplementedError("Only support l = s..h.")
    left_mats = [DFTIO2ABACUS[l] for l in l_lefts]
    right_mats = [DFTIO2ABACUS[l] for l in l_rights]
    left = block_diag(*left_mats) if left_mats else np.eye(0, dtype=np.float32)
    right = block_diag(*right_mats) if right_mats else np.eye(0, dtype=np.float32)
    return left @ mat @ right.T


def write_abacus_csr_format(matrix_dict, matrix_symbol, output_path, step=0):
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
                np.savetxt(f, sparse_mat.data.reshape(1, -1), fmt='%.8e')
                np.savetxt(f, sparse_mat.indices.reshape(1, -1), fmt='%d')
                np.savetxt(f, sparse_mat.indptr.reshape(1, -1), fmt='%d')
            else:
                f.write("\n\n\n")
    # print(f"Wrote {num_blocks} blocks to {output_path}")


def write_blocks_to_abacus_csr(atomic_numbers, basis_dict, blocks_dict, matrix_symbol, output_path, step=0):
    """
    Entry function:
      atomic_numbers: per-site Z array-like
      basis_dict: parse_orbital_files result
      blocks_dict: mapping 'i_j_Rx_Ry_Rz' -> small block (DFTIO ordering)
      matrix_symbol: 'H'/'S'/'D'
    """
    atomic_numbers = np.asarray(atomic_numbers, dtype=int)
    if atomic_numbers.size == 0:
        raise ValueError("empty atomic_numbers")

    # choose factor
    factor = H_FACTOR if str(matrix_symbol).upper() == 'H' else 1.0

    # element -> l-list
    element_l_lists = {}
    for Z in np.unique(atomic_numbers):
        basis_str = find_basis_for_Z_or_symbol(basis_dict, int(Z))
        if basis_str is None:
            element_l_lists[int(Z)] = [0]
        else:
            ll = parse_basis_to_l_list(basis_str)
            element_l_lists[int(Z)] = ll if ll else [0]

    # site norbits
    site_norbits = np.array([sum(2 * l + 1 for l in element_l_lists[int(Z)]) for Z in atomic_numbers], dtype=int)
    site_norbits_cumsum = np.cumsum(site_norbits)
    norbits = int(site_norbits_cumsum[-1])

    # aggregate COO data per R
    r_vector_coo = defaultdict(lambda: {'data': [], 'rows': [], 'cols': []})

    for raw_key, small_block in blocks_dict.items():
        key = raw_key.decode() if isinstance(raw_key, (bytes, bytearray)) else str(raw_key)
        m = KEY_RE.match(key)
        if not m:
            # skip unparseable keys
            continue
        i_site = int(m.group(1)); j_site = int(m.group(2))
        Rx = int(m.group(3)); Ry = int(m.group(4)); Rz = int(m.group(5))
        r_str = f"{Rx}_{Ry}_{Rz}"

        # l-lists
        l_lefts = element_l_lists[int(atomic_numbers[i_site])]
        l_rights = element_l_lists[int(atomic_numbers[j_site])]

        # get ndarray (support sparse objects)
        if hasattr(small_block, "toarray"):
            block_arr = small_block.toarray()
        elif "torch" in str(type(small_block)):
            if small_block.is_cuda:
                block_arr = small_block.detach().cpu().numpy()
            else:
                block_arr = small_block.detach().numpy()
        else:
            block_arr = np.asarray(small_block)
        if block_arr.size == 0:
            continue

        # transform DFTIO -> ABACUS
        transformed = transform_2_ABACUS(block_arr.astype(np.float32), l_lefts, l_rights)

        # offsets
        row_offset = int(site_norbits_cumsum[i_site] - site_norbits[i_site])
        col_offset = int(site_norbits_cumsum[j_site] - site_norbits[j_site])

        coo = coo_matrix(transformed)
        if coo.nnz == 0:
            continue

        # apply factor (H vs others)
        r_vector_coo[r_str]['data'].append((coo.data.astype(np.float32) / factor))
        r_vector_coo[r_str]['rows'].append((coo.row + row_offset).astype(int))
        r_vector_coo[r_str]['cols'].append((coo.col + col_offset).astype(int))

    # build final CSR dict
    reassembled = {}
    for r_str, parts in r_vector_coo.items():
        if not parts['data']:
            full = csr_matrix((norbits, norbits), dtype=np.float32)
        else:
            data = np.concatenate(parts['data']).astype(np.float32)
            rows = np.concatenate(parts['rows']).astype(int)
            cols = np.concatenate(parts['cols']).astype(int)
            full = csr_matrix((data, (rows, cols)), shape=(norbits, norbits))
        reassembled[r_str] = full

    write_abacus_csr_format(reassembled, matrix_symbol, output_path, step=step)
    return reassembled, norbits


# demo main
if __name__ == "__main__":
    LMDB_PATH = r'E:\deeptb\large_DeepTB\0909\0910_lmdb\train\data.28400.lmdb'
    ORBITAL_PATH = r'E:\deeptb\basis_set_test\production_use_dzp\orb_upf\public'

    from dprep.dptb_dpdispatcher import parse_orbital_files
    _, basis_dict = parse_orbital_files(ORBITAL_PATH)

    env = lmdb.open(LMDB_PATH, readonly=True, lock=False)
    with env.begin() as txn:
        rec = txn.get((0).to_bytes(length=4, byteorder='big'))
        if rec is None:
            raise RuntimeError("No record at index 0")
        data = pickle.loads(rec)
    env.close()

    atomic_numbers = np.array(data['atomic_numbers'], dtype=int)

    if 'hamiltonian' in data and data['hamiltonian']:
        write_blocks_to_abacus_csr(
            atomic_numbers=atomic_numbers,
            basis_dict=basis_dict,
            blocks_dict=data['hamiltonian'],
            matrix_symbol='H',
            output_path='data-HR-sparse_SPIN0.csr',
            step=0
        )
    else:
        print("No hamiltonian in record 0.")
