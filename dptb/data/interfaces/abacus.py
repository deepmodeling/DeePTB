# Modified from script 'abasus_get_data.py' for interface from ABACUS to DeepH-pack
# To use this script, please add 'out_mat_hs2    1' in ABACUS INPUT File
# Current version is capable of coping with f-orbitals

import os
import glob
import json
import re
from collections import Counter
from tqdm import tqdm

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
import h5py
import ase

orbitalId = {0:'s',1:'p',2:'d',3:'f',4:'g',5:'h'}
Bohr2Ang = 0.529177249


class OrbAbacus2DeepTB:
    def __init__(self):
        self.Us_abacus2deeptb = {}
        self.Us_abacus2deeptb[0] = np.eye(1)
        self.Us_abacus2deeptb[1] = np.eye(3)[[2, 0, 1]]            # 0, 1, -1 -> -1, 0, 1
        self.Us_abacus2deeptb[2] = np.eye(5)[[4, 2, 0, 1, 3]]      # 0, 1, -1, 2, -2 -> -2, -1, 0, 1, 2
        self.Us_abacus2deeptb[3] = np.eye(7)[[6, 4, 2, 0, 1, 3, 5]] # -3,-2,-1,0,1,2,3
        self.Us_abacus2deeptb[4] = np.eye(9)[[8, 6, 4, 2, 0, 1, 3, 5, 7]]
        self.Us_abacus2deeptb[5] = np.eye(11)[[10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9]]

        minus_dict = {
            1: [0, 2],
            2: [1, 3],
            3: [0, 2, 4, 6],
            4: [1, 7, 3, 5],
            5: [0, 8, 2, 6, 4]
        }

        for k, v in minus_dict.items():
            self.Us_abacus2deeptb[k][v] *= -1  # add phase (-1)^m

    def get_U(self, l):
        if l > 5:
            raise NotImplementedError("Only support l = s, p, d, f, g, h.")
        return self.Us_abacus2deeptb[l]

    def transform(self, mat, l_lefts, l_rights):
        block_lefts = block_diag(*[self.get_U(l_left) for l_left in l_lefts])
        block_rights = block_diag(*[self.get_U(l_right) for l_right in l_rights])
        return block_lefts @ mat @ block_rights.T
    
def recursive_parse(input_path, 
                    preprocess_dir, 
                    data_name="OUT.ABACUS",
                    parse_Hamiltonian=False, 
                    parse_overlap=False,
                    parse_DM=False, 
                    parse_eigenvalues=False,
                    prefix="data"):
    """
    Parse ABACUS single point SCF calculation outputs.
    Input:
    `input_dir`: target dictionary(ies) containing "OUT.ABACUS" folder.
                 can be wildcard characters or a string list.
    `preprocess_dir`: output dictionary of all processed data files.
    `data_name`: output dictionary name of ABACUS, by default "OUT.ABACUS".
    `only_overlap`: usually `False`. 
                    set to `True` if the calculation job is getting overlap matrix ONLY.
    `parse_Hamiltonian`: determine whether parsing the Hamiltonian `.csr` file or not.
    `add_overlap`: determine whether parsing the overlap `.csr` file or not.
                   `parse_Hamiltonian` must be true to add overlap.
    `parse_eigenvalues`: determine whether parsing `kpoints.dat` and `BAND_1.dat` or not.
                         that is, the k-points will always be loaded with bands.
    `prefix`: prefix of the processed data folders' names. 
    """
    if isinstance(input_path, list) and all(isinstance(item, str) for item in input_path):
        input_path = input_path
    else:
        input_path = glob.glob(input_path)
    preprocess_dir = os.path.abspath(preprocess_dir)
    os.makedirs(preprocess_dir, exist_ok=True)
    # h5file_names = []

    folders = [item for item in input_path if os.path.isdir(item)]

    with tqdm(total=len(folders)) as pbar:
        for index, folder in enumerate(folders):
            datafiles = os.listdir(folder)
            if data_name in datafiles:
                # The follwing `if` block is used by us only.
                if os.path.exists(os.path.join(folder, data_name, "hscsr.tgz")):
                    os.system("cd "+os.path.join(folder, data_name) + " && tar -zxvf hscsr.tgz && mv OUT.ABACUS/* ./")
                try:
                    _abacus_parse(folder, 
                                  os.path.join(preprocess_dir, f"{prefix}.{index}"), 
                                  data_name,
                                  get_Ham=parse_Hamiltonian,
                                  get_DM=parse_DM,
                                  get_overlap=parse_overlap, 
                                  get_eigenvalues=parse_eigenvalues)
                    #h5file_names.append(os.path.join(file, "AtomicData.h5"))
                    pbar.update(1)
                except Exception as e:
                    print(f"Error in {folder}/{data_name}: {e}")
                    continue
    #return h5file_names

def _abacus_parse(input_path, 
                  output_path, 
                  data_name, 
                  only_S=False, 
                  get_Ham=False,
                  get_DM=False,
                  get_overlap=False, 
                  get_eigenvalues=False):
    
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)

    def find_target_line(f, target):
        line = f.readline()
        while line:
            if target in line:
                return line
            line = f.readline()
        return None
    if only_S:
        log_file_name = "running_get_S.log"
    else:
        log_file_name = "running_scf.log"

    with open(os.path.join(input_path, data_name, log_file_name), 'r') as f_chk:
        lines = f_chk.readlines()
        if not lines or " Total  Time  :" not in lines[-1]:
            raise ValueError(f"Job is not normal ending!")

    with open(os.path.join(input_path, data_name, log_file_name), 'r') as f:
        f.readline()
        line = f.readline()
        # assert "WELCOME TO ABACUS" in line
        assert find_target_line(f, "READING UNITCELL INFORMATION") is not None, 'Cannot find "READING UNITCELL INFORMATION" in log file'
        num_atom_type = int(f.readline().split()[-1])

        assert find_target_line(f, "lattice constant (Bohr)") is not None
        lattice_constant = float(f.readline().split()[-1]) # unit is Angstrom, didn't read (Bohr) here.

        site_norbits_dict = {}
        orbital_types_dict = {}
        for index_type in range(num_atom_type):
            tmp = find_target_line(f, "READING ATOM TYPE")
            assert tmp is not None, 'Cannot find "ATOM TYPE" in log file'
            assert tmp.split()[-1] == str(index_type + 1)
            if tmp is None:
                raise Exception(f"Cannot find ATOM {index_type} in {log_file_name}")

            line = f.readline()
            assert "atom label =" in line
            atom_label = line.split()[-1]
            atom_label = ''.join(re.findall(r'[A-Za-z]', atom_label))
            assert atom_label in ase.data.atomic_numbers, "Atom label should be in periodic table"
            atom_type = ase.data.atomic_numbers[atom_label]

            current_site_norbits = 0
            current_orbital_types = []
            while True:
                line = f.readline()
                if "number of zeta" in line:
                    tmp = line.split()
                    L = int(tmp[0][2:-1])
                    num_L = int(tmp[-1])
                    current_site_norbits += (2 * L + 1) * num_L
                    current_orbital_types.extend([L] * num_L)
                else:
                    break
            site_norbits_dict[atom_type] = current_site_norbits
            orbital_types_dict[atom_type] = current_orbital_types

        #print(orbital_types_dict)

        line = find_target_line(f, "TOTAL ATOM NUMBER")
        assert line is not None, 'Cannot find "TOTAL ATOM NUMBER" in log file'
        nsites = int(line.split()[-1])
        
        line = find_target_line(f, " COORDINATES")
        assert line is not None, 'Cannot find "DIRECT COORDINATES" or "CARTESIAN COORDINATES" in log file'
        if "DIRECT" in line:
            coords_type = "direct" 
        elif "CARTESIAN" in line:
            coords_type = "cartesian" 
        else:
            raise ValueError('Cannot find "DIRECT COORDINATES" or "CARTESIAN COORDINATES" in log file')

        assert "atom" in f.readline()
        frac_coords = np.zeros((nsites, 3))
        site_norbits = np.zeros(nsites, dtype=int)
        element = np.zeros(nsites, dtype=int)
        for index_site in range(nsites):
            line = f.readline()
            tmp = line.split()
            assert "tau" in tmp[0]
            atom_label = ''.join(re.findall(r'[A-Za-z]', tmp[0][5:]))
            assert atom_label in ase.data.atomic_numbers, "Atom label should be in periodic table"
            element[index_site] = ase.data.atomic_numbers[atom_label]
            site_norbits[index_site] = site_norbits_dict[element[index_site]]
            frac_coords[index_site, :] = np.array(tmp[1:4])
        norbits = int(np.sum(site_norbits))
        site_norbits_cumsum = np.cumsum(site_norbits)

        assert find_target_line(f, "Lattice vectors: (Cartesian coordinate: in unit of a_0)") is not None
        lattice = np.zeros((3, 3))
        for index_lat in range(3):
            lattice[index_lat, :] = np.array(f.readline().split())
        if coords_type == "cartesian":
            frac_coords = frac_coords @ np.matrix(lattice).I  # get frac_coords anyway
        lattice = lattice * lattice_constant

        if get_Ham is False and get_overlap is True:
            spinful = False
        else:
            line = find_target_line(f, "NSPIN")
            assert line is not None, 'Cannot find "NSPIN" in log file'
            if "NSPIN == 1" in line:
                spinful = False
            elif "NSPIN == 4" in line:
                spinful = True
            else:
                raise ValueError(f'{line} is not supported')

    np.savetxt(os.path.join(output_path, "cell.dat"), lattice)
    np.savetxt(os.path.join(output_path, "rcell.dat"), np.linalg.inv(lattice) * 2 * np.pi)
    cart_coords = frac_coords @ lattice
    np.savetxt(os.path.join(output_path, "positions.dat").format(output_path), cart_coords)
    np.savetxt(os.path.join(output_path, "atomic_numbers.dat"), element, fmt='%d')
    #info = {'nsites' : nsites, 'isorthogonal': False, 'isspinful': spinful, 'norbits': norbits}
    #with open('{}/info.json'.format(output_path), 'w') as info_f:
    #    json.dump(info, info_f)
    with open(os.path.join(output_path, "basis.dat"), 'w') as f:
        for atomic_number in element:
            counter = Counter(orbital_types_dict[atomic_number])
            num_orbs = [counter[i] for i in range(4)] # s, p, d, f
            for index_l, l in enumerate(num_orbs):
                if l == 0:  # no this orbit
                    continue
                if index_l == 0:
                    f.write(f"{l}{orbitalId[index_l]}")
                else:
                    f.write(f"{l}{orbitalId[index_l]}")
            f.write('\n')
    atomic_basis = {}
    for atomic_number, orbitals in orbital_types_dict.items():
        atomic_basis[ase.data.chemical_symbols[atomic_number]] = orbitals

    U_orbital = OrbAbacus2DeepTB()
    def parse_matrix(matrix_path, factor, spinful=False):
        matrix_dict = dict()
        with open(matrix_path, 'r') as f:
            line = f.readline() # read "Matrix Dimension of ..."
            if not "Matrix Dimension of" in line:
                line = f.readline() # ABACUS >= 3.0
                assert "Matrix Dimension of" in line
            f.readline() # read "Matrix number of ..."
            norbits = int(line.split()[-1])
            for line in f:
                line1 = line.split()
                if len(line1) == 0:
                    break
                num_element = int(line1[3])
                if num_element != 0:
                    R_cur = np.array(line1[:3]).astype(int)
                    line2 = f.readline().split()
                    line3 = f.readline().split()
                    line4 = f.readline().split()
                    if not spinful:
                        hamiltonian_cur = csr_matrix((np.array(line2).astype(float), np.array(line3).astype(int),
                                                        np.array(line4).astype(int)), shape=(norbits, norbits)).toarray()
                    else:
                        line2 = np.char.replace(line2, '(', '')
                        line2 = np.char.replace(line2, ')', 'j')
                        line2 = np.char.replace(line2, ',', '+')
                        line2 = np.char.replace(line2, '+-', '-')
                        hamiltonian_cur = csr_matrix((np.array(line2).astype(np.complex128), np.array(line3).astype(int),
                                                    np.array(line4).astype(int)), shape=(norbits, norbits)).toarray()
                    for index_site_i in range(nsites):
                        for index_site_j in range(nsites):
                            key_str = f"{index_site_i + 1}_{index_site_j + 1}_{R_cur[0]}_{R_cur[1]}_{R_cur[2]}"
                            mat = hamiltonian_cur[(site_norbits_cumsum[index_site_i]
                                                    - site_norbits[index_site_i]) * (1 + spinful):
                                                    site_norbits_cumsum[index_site_i] * (1 + spinful),
                                    (site_norbits_cumsum[index_site_j] - site_norbits[index_site_j]) * (1 + spinful):
                                    site_norbits_cumsum[index_site_j] * (1 + spinful)]
                            if abs(mat).max() < 1e-10:
                                continue
                            if not spinful:
                                mat = U_orbital.transform(mat, orbital_types_dict[element[index_site_i]],
                                                            orbital_types_dict[element[index_site_j]])
                            else:
                                mat = mat.reshape((site_norbits[index_site_i], 2, site_norbits[index_site_j], 2))
                                mat = mat.transpose((1, 0, 3, 2)).reshape((2 * site_norbits[index_site_i],
                                                                        2 * site_norbits[index_site_j]))
                                mat = U_orbital.transform(mat, orbital_types_dict[element[index_site_i]] * 2,
                                                        orbital_types_dict[element[index_site_j]] * 2)
                            matrix_dict[key_str] = mat * factor
        return matrix_dict, norbits

    if get_Ham:
        hamiltonian_dict, tmp = parse_matrix(
            os.path.join(input_path, data_name, "data-HR-sparse_SPIN0.csr"), 13.605698, # Ryd2eV
            spinful=spinful)
        assert tmp == norbits * (1 + spinful)

        with h5py.File(os.path.join(output_path, "hamiltonians.h5"), 'w') as fid:
            # creating a default group here adapting to the format used in DefaultDataset.
            # by the way DefaultDataset loading h5 file, the index should be "1" here.
            default_group = fid.create_group("1")
            for key_str, value in hamiltonian_dict.items():
                default_group[key_str] = value

    if get_overlap:
        overlap_dict, tmp = parse_matrix(os.path.join(input_path, data_name, "data-SR-sparse_SPIN0.csr"), 1,
                                            spinful=spinful)
        assert tmp == norbits * (1 + spinful)
        if spinful:
            overlap_dict_spinless = {}
            for k, v in overlap_dict.items():
                overlap_dict_spinless[k] = v[:v.shape[0] // 2, :v.shape[1] // 2].real
            overlap_dict_spinless, overlap_dict = overlap_dict, overlap_dict_spinless

        with h5py.File(os.path.join(output_path, "overlaps.h5"), 'w') as fid:
            default_group = fid.create_group("1")
            for key_str, value in overlap_dict.items():
                default_group[key_str] = value
            
    if get_DM:
        DM_dict, tmp = parse_matrix(os.path.join(input_path, data_name, "data-DMR-sparse_SPIN0.csr"), 1,
                                            spinful=spinful)
        assert tmp == norbits * (1 + spinful)
        # if spinful:
        #     DM_dict_spinless = {}
        #     for k, v in overlap_dict.items():
        #         overlap_dict_spinless[k] = v[:v.shape[0] // 2, :v.shape[1] // 2].real
        #     overlap_dict_spinless, overlap_dict = overlap_dict, overlap_dict_spinless

        with h5py.File(os.path.join(output_path, "DM.h5"), 'w') as fid:
            default_group = fid.create_group("1")
            for key_str, value in overlap_dict.items():
                default_group[key_str] = value

    if get_eigenvalues:
        kpts = []
        with open(os.path.join(input_path, data_name, "kpoints"), "r") as f:
            nkstot = f.readline().strip().split()[-1]
            f.readline()
            for _ in range(int(nkstot)):
                line = f.readline()
                kpt = []
                line = line.strip().split()
                kpt.extend([float(line[1]), float(line[2]), float(line[3])])
                kpts.append(kpt)
        kpts = np.array(kpts)

        with open(os.path.join(input_path, data_name, "BANDS_1.dat"), "r") as file:
            band_lines = file.readlines()
        band = []
        for line in band_lines:
            values = line.strip().split()
            eigs = [float(value) for value in values[1:]]
            band.append(eigs)
        band = np.array(band)

        assert len(band) == len(kpts)
        np.save(os.path.join(output_path, "kpoints.npy"), kpts)
        np.save(os.path.join(output_path, "eigenvalues.npy"), band)

    #with h5py.File(os.path.join(output_path, "AtomicData.h5"), "w") as f:
    #    f["cell"] = lattice
    #    f["pos"] = cart_coords
    #    f["atomic_numbers"] = element
    #    basis = f.create_group("basis")
    #    for key, value in atomic_basis.items():
    #        basis[key] = value
    #    if get_Ham:
    #        f["hamiltonian_blocks"] = h5py.ExternalLink("hamiltonians.h5", "/")
    #        if add_overlap:
    #            f["overlap_blocks"] = h5py.ExternalLink("overlaps.h5", "/")
    #        # else:
    #        #     f["overlap_blocks"] = False
    #    # else:
    #    #     f["hamiltonian_blocks"] = False
    #    if get_eigenvalues:
    #        f["kpoints"] = kpts
    #        f["eigenvalues"] = band
    #    # else:
    #    #     f["kpoint"] = False
    #    #     f["eigenvalue"] = False
