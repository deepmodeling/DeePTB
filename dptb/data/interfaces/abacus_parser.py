from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
import re
from tqdm import tqdm
from collections import Counter
from dptb.utils.constants import orbitalId, Bohr2Ang, ABACUS2DeePTB
import ase
import dpdata
import os
import numpy as np
from .parse import Parser, find_target_line
from .. import _keys

class AbacusParser(Parser):
    def __init__(
            self,
            root,
            prefix,
            ):
        super(AbacusParser, self).__init__(root, prefix)
        self.raw_sys = [dpdata.LabeledSystem(self.raw_datas[idx], fmt='abacus/'+self.get_mode(idx)) for idx in range(len(self.raw_datas))]
        
    def get_structure(self, idx):
        sys = self.raw_sys[idx]
        
        structure = {
            _keys.ATOMIC_NUMBERS_KEY: np.array([ase.atom.atomic_numbers[i] for i in sys.data["atom_names"]], dtype=np.int32)[sys.data["atom_types"]],
            _keys.PBC_KEY: np.array([True, True, True]) # abacus does not allow non-pbc structure
        }
        structure[_keys.POSITIONS_KEY] = sys.data["coords"].astype(np.float32)
        structure[_keys.CELL_KEY] = sys.data["cells"].astype(np.float32)

        return structure
    
    def get_mode(self, idx):
        with open(os.path.join(self.raw_datas[idx], "OUT.ABACUS", "INPUT"), 'r') as f:
            line = find_target_line(f, "calculation")
            assert line is not None, 'Cannot find "MODE" in log file'
            mode = line.split()[1]
            f.close()

        return mode
    
    def get_eigenvalue(self, idx):
        path = self.raw_datas[idx]
        mode = self.get_mode(idx)
        if mode=="scf":
            assert os.path.exists(os.path.join(path, "OUT.ABACUS", "BANDS_1.dat"))
            eigs = np.loadtxt(os.path.join(path, "OUT.ABACUS", "BANDS_1.dat"))[np.newaxis, :, 2:]
            assert os.path.exists(os.path.join(path, "OUT.ABACUS", "kpoints"))
            kpts = []
            with open(os.path.join(path, "OUT.ABACUS", "kpoints"), "r") as f:
                line = find_target_line(f, "nkstot now")
                nkstot = line.strip().split()[-1]
                line = find_target_line(f, " KPOINTS     DIRECT_X")
                for _ in range(int(nkstot)):
                    line = f.readline()
                    kpt = []
                    line = line.strip().split()
                    kpt.extend([float(line[1]), float(line[2]), float(line[3])])
                    kpts.append(kpt)
                kpts = np.array(kpts)
        elif mode == "md" or mode == "relax":
            raise NotImplementedError("output eigenvalues from MD trajectory is not supported by ABACUS.")
        
        else:
            raise NotImplementedError("mode {} is not supported.".format(mode))
            
        return {_keys.ENERGY_EIGENVALUE_KEY: eigs, _keys.KPOINT_KEY: kpts}
    
    def get_basis(self, idx):
        mode = self.get_mode(idx)
        logfile = "running_"+mode+".log"
        sys = self.raw_sys[idx]
        with open(os.path.join(self.raw_datas[idx], "OUT.ABACUS", logfile), 'r') as f:
            orbital_types_dict = {}
            for index_type in range(len(sys.data["atom_numbs"])):
                tmp = find_target_line(f, "READING ATOM TYPE")
                assert tmp is not None, 'Cannot find "ATOM TYPE" in log file'
                assert tmp.split()[-1] == str(index_type + 1)
                if tmp is None:
                    raise Exception(f"Cannot find ATOM {index_type} in {logfile}")

                line = f.readline()
                assert "atom label =" in line
                atom_label = line.split()[-1]
                atom_label = ''.join(re.findall(r'[A-Za-z]', atom_label))
                assert atom_label in ase.data.atomic_numbers, "Atom label should be in periodic table"

                current_orbital_types = []
                while True:
                    line = f.readline()
                    if "number of zeta" in line:
                        tmp = line.split()
                        L = int(tmp[0][2:-1])
                        num_L = int(tmp[-1])
                        current_orbital_types.extend([L] * num_L)
                    else:
                        break
                orbital_types_dict[atom_label] = current_orbital_types
        basis = {}
        for k,v in orbital_types_dict.items():
            counter = Counter(v)
            basis[k] = [str(counter[l])+orbitalId[l] for l in range(max(counter.keys())) if counter.get(l, 0) > 0]
            basis[k] = "".join(basis[k])
        
        return basis
    
    def get_blocks(self, idx, hamiltonian=True, overlap=False, density_matrix=False):
        mode = self.get_mode(idx)
        logfile = "running_"+mode+".log"
        hamiltonian_dict, overlap_dict, density_matrix_dict = None, None, None
        sys = self.raw_sys[idx]
        nsites = sys.data["atom_types"].shape[0]
        with open(os.path.join(self.raw_datas[idx], "OUT.ABACUS", logfile), 'r') as f:
            site_norbits_dict = {}
            orbital_types_dict = {}
            for index_type in range(len(sys.data["atom_numbs"])):
                tmp = find_target_line(f, "READING ATOM TYPE")
                assert tmp is not None, 'Cannot find "ATOM TYPE" in log file'
                assert tmp.split()[-1] == str(index_type + 1)
                if tmp is None:
                    raise Exception(f"Cannot find ATOM {index_type} in {logfile}")

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


            line = find_target_line(f, " COORDINATES")
            assert "atom" in f.readline()
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

            if hamiltonian is False and overlap is True:
                spinful = False
            else:
                line = find_target_line(f, "nspin")
                if line is None:
                    line = find_target_line(f, "NSPIN")
                assert line is not None, 'Cannot find "NSPIN" in log file'
                if "NSPIN == 1" or "npin = 1" in line:
                    spinful = False
                elif "NSPIN == 4" or "nspin = 4" in line:
                    spinful = True
                else:
                    raise ValueError(f'{line} is not supported')

        if mode == "scf":
            if hamiltonian:
                hamiltonian_dict, tmp = self.parse_matrix(
                    matrix_path=os.path.join(self.raw_datas[idx], "OUT.ABACUS", "data-HR-sparse_SPIN0.csr"), 
                    nsites=nsites,
                    site_norbits=site_norbits,
                    orbital_types_dict=orbital_types_dict,
                    element=element,
                    factor=13.605698, # Ryd2eV
                    spinful=spinful
                    )
                assert tmp == int(np.sum(site_norbits)) * (1 + spinful)
                hamiltonian_dict = [hamiltonian_dict]
            
            if overlap:
                overlap_dict, tmp = self.parse_matrix(
                    matrix_path=os.path.join(self.raw_datas[idx], "OUT.ABACUS", "data-SR-sparse_SPIN0.csr"), 
                    nsites=nsites,
                    site_norbits=site_norbits,
                    orbital_types_dict=orbital_types_dict,
                    element=element,
                    factor=1,
                    spinful=spinful
                    )
                assert tmp == int(np.sum(site_norbits)) * (1 + spinful)

                if spinful:
                    overlap_dict_spinless = {}
                    for k, v in overlap_dict.items():
                        overlap_dict_spinless[k] = v[:v.shape[0] // 2, :v.shape[1] // 2].real
                    overlap_dict_spinless, overlap_dict = overlap_dict, overlap_dict_spinless

                overlap_dict = [overlap_dict]

            if density_matrix:
                density_matrix_dict, tmp = self.parse_matrix(
                    matrix_path=os.path.join(self.raw_datas[idx], "OUT.ABACUS", "data-DMR-sparse_SPIN0.csr"), 
                    nsites=nsites,
                    site_norbits=site_norbits,
                    orbital_types_dict=orbital_types_dict,
                    element=element,
                    factor=1,
                    spinful=spinful
                    )
                assert tmp == int(np.sum(site_norbits)) * (1 + spinful)

                density_matrix_dict = [density_matrix_dict]

        elif mode == "md":
            if hamiltonian:
                hamiltonian_dict = []
                for i in range(sys.get_nframes()):
                    hamil, tmp = self.parse_matrix(
                        matrix_path=os.path.join(self.raw_datas[idx], "OUT.ABACUS", "matrix/"+str(i)+"_data-HR-sparse_SPIN0.csr"), 
                        nsites=nsites,
                        site_norbits=site_norbits,
                        orbital_types_dict=orbital_types_dict,
                        element=element,
                        factor=13.605698, # Ryd2eV
                        spinful=spinful
                        )
                    assert tmp == int(np.sum(site_norbits)) * (1 + spinful)
                    hamiltonian_dict.append(hamil)

            if overlap:
                overlap_dict = []
                for i in range(sys.get_nframes()):
                    ovp, tmp = self.parse_matrix(
                        matrix_path=os.path.join(self.raw_datas[idx], "OUT.ABACUS", "matrix/"+str(i)+"_data-SR-sparse_SPIN0.csr"), 
                        nsites=nsites,
                        site_norbits=site_norbits,
                        orbital_types_dict=orbital_types_dict,
                        element=element,
                        factor=1,
                        spinful=spinful
                        )
                    assert tmp == int(np.sum(site_norbits)) * (1 + spinful)

                    if spinful:
                        ovp_spinless = {}
                        for k, v in ovp.items():
                            ovp_spinless[k] = v[:v.shape[0] // 2, :v.shape[1] // 2].real
                    overlap_dict.append(ovp_spinless)
            
            if density_matrix:
                density_matrix_dict = []
                for i in range(sys.get_nframes()):
                    dm, tmp = self.parse_matrix(
                        matrix_path=os.path.join(self.raw_datas[idx], "OUT.ABACUS", "matrix/"+str(i)+"_data-DMR-sparse_SPIN0.csr"), 
                        nsites=nsites,
                        site_norbits=site_norbits,
                        orbital_types_dict=orbital_types_dict,
                        element=element,
                        factor=1,
                        spinful=spinful
                        )
                    assert tmp == int(np.sum(site_norbits)) * (1 + spinful)
                    density_matrix_dict.append(dm)
        else:
            raise NotImplementedError("mode {} is not supported.".format(mode))
        
        return hamiltonian_dict, overlap_dict, density_matrix_dict

    def parse_matrix(self, matrix_path, nsites, site_norbits, orbital_types_dict, element, factor, spinful=False):
        site_norbits_cumsum = np.cumsum(site_norbits)
        norbits = int(np.sum(site_norbits))
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
                        hamiltonian_cur = csr_matrix((np.array(line2).astype(np.float32), np.array(line3).astype(int),
                                                        np.array(line4).astype(np.int32)), shape=(norbits, norbits), dtype=np.float32).toarray()
                    else:
                        line2 = np.char.replace(line2, '(', '')
                        line2 = np.char.replace(line2, ')', 'j')
                        line2 = np.char.replace(line2, ',', '+')
                        line2 = np.char.replace(line2, '+-', '-')
                        hamiltonian_cur = csr_matrix((np.array(line2).astype(np.complex64), np.array(line3).astype(int),
                                                    np.array(line4).astype(np.int32)), shape=(norbits, norbits), dtype=np.complex64).toarray()
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
                                mat = self.transform(mat, orbital_types_dict[element[index_site_i]],
                                                            orbital_types_dict[element[index_site_j]])
                            else:
                                mat = mat.reshape((site_norbits[index_site_i], 2, site_norbits[index_site_j], 2))
                                mat = mat.transpose((1, 0, 3, 2)).reshape((2 * site_norbits[index_site_i],
                                                                        2 * site_norbits[index_site_j]))
                                mat = self.transform(mat, orbital_types_dict[element[index_site_i]] * 2,
                                                        orbital_types_dict[element[index_site_j]] * 2)
                            matrix_dict[key_str] = mat * factor
        return matrix_dict, norbits
    
    def transform(self, mat, l_lefts, l_rights):

        if max(*l_lefts, *l_rights) > 5:
            raise NotImplementedError("Only support l = s, p, d, f, g, h.")
        block_lefts = block_diag(*[ABACUS2DeePTB[l_left] for l_left in l_lefts])
        block_rights = block_diag(*[ABACUS2DeePTB[l_right] for l_right in l_rights])

        return block_lefts @ mat @ block_rights.T

        
