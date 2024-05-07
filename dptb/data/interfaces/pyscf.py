import os
import h5py
import glob
from dptb.utils.constants import orbitalId, Bohr2Ang, PYSCF2DeePTB
from scipy.linalg import block_diag
import scipy.constants as const
import numpy as np
from dptb.utils.constants import atomic_num_dict, atomic_num_dict_r, anglrMId
from dptb.utils.symbol2ID import symbol2ID
import argparse


# 使用 scipy.constants 获取玻尔半径的值，单位是米
bohr_radius_m = const.physical_constants["Bohr radius"][0]
# 将玻尔半径从米转换为埃（1埃 = 1e-10 米）
bohr_to_angstrom = bohr_radius_m * 1e10


class OrbDFT2DeepTB:
    def __init__(self, DFT2DeePTB = None):
        
        if DFT2DeePTB is None:
            DFT2DeePTB = PYSCF2DeePTB
        self.Us_DFT2DeePTB = DFT2DeePTB

    def get_U(self, l):
        if l > 5:
            raise NotImplementedError("Only support l = s, p, d, f, g, h.")
        return self.Us_DFT2DeePTB[l]

    def transform(self, mat, l_lefts, l_rights):
        block_lefts = block_diag(*[self.get_U(l_left) for l_left in l_lefts])
        block_rights = block_diag(*[self.get_U(l_right) for l_right in l_rights])
        return block_lefts @ mat @ block_rights.T
    

pyscf_basis = {
    "H":["3s","2p","1d"],
    "C":["4s","3p","2d","1f"],
    "N":["4s","3p","2d","1f"],
    "O":["4s","3p","2d","1f"],
    "F":["4s","3p","2d","1f"]
}


chem_symbols = list(symbol2ID.keys())

def _chkfile_parse(chkfile, 
                   site_norbits_dict,
                   orbital_types_dict,
                   get_DM=True):
    

    h5dat = h5py.File(chkfile, 'r')
    atom_numbers = h5dat['atomic_numbers'][:]
    coords = h5dat['coords'][:] *  bohr_to_angstrom
    
    nsites = len(atom_numbers)
    assert nsites == len(coords)

    U_orbital = OrbDFT2DeepTB(DFT2DeePTB = PYSCF2DeePTB)

    if get_DM:
        dm = h5dat['dm'][:]
        R_cur = [0,0,0]
        matrix_dict = dict()

        for index_site_i in range(nsites):
            for index_site_j in range(nsites):
                key_str = f"{index_site_i + 1}_{index_site_j + 1}_{R_cur[0]}_{R_cur[1]}_{R_cur[2]}"

                norb_i = site_norbits_dict[atomic_num_dict_r[atom_numbers[index_site_i]]]
                norb_j = site_norbits_dict[atomic_num_dict_r[atom_numbers[index_site_j]]]

                ist = int(np.sum(np.array([site_norbits_dict[atomic_num_dict_r[atom_numbers[_ii]]] for _ii in range(index_site_i)])))
                jst = int(np.sum(np.array([site_norbits_dict[atomic_num_dict_r[atom_numbers[_jj]]] for _jj in range(index_site_j)])))

                mat = dm[ist:ist+norb_i, jst:jst+norb_j]
                if abs(mat).max() < 1e-10:
                    continue
                
                mat = U_orbital.transform(mat, orbital_types_dict[atomic_num_dict_r[atom_numbers[index_site_i]]], 
                                            orbital_types_dict[atomic_num_dict_r[atom_numbers[index_site_j]]])

                matrix_dict[key_str] = mat  
    else:
        raise NotImplementedError("Only support get_DM=True.")
    
    return coords, atom_numbers, matrix_dict

def _pyscf_parse_qm9(input_path, 
                     output_path, 
                     data_name=None, 
                     get_DM=True):
    
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)

    if data_name is None:
        data_name =  chem_symbols
    elif isinstance(data_name, str):
        data_name = [data_name]
    elif isinstance(data_name, list):
        for name in data_name:
            if name not in chem_symbols:
                raise ValueError(f"{name} is not a valid chemical symbol.")
    else:
        raise ValueError("data_name must be a string or a list of strings.")
    
    site_norbits_dict = {}
    orbital_types_dict = {}
    for ia in pyscf_basis.keys():
        basis = pyscf_basis[ia]
        current_site_norbits = 0
        current_orbital_types = []
        for iorb in basis:
            assert len(iorb) == 2
            l = anglrMId[iorb[1]]
            num_l = int(iorb[0])
            current_site_norbits += num_l * (2*l + 1)
            current_orbital_types.extend([l] * num_l)
        site_norbits_dict[ia] = current_site_norbits
        orbital_types_dict[ia] = current_orbital_types


    U_orbital = OrbDFT2DeepTB(DFT2DeePTB = PYSCF2DeePTB)
    for isymbol  in data_name:
        iID_lists  = symbol2ID[isymbol]
        out = os.path.join(output_path, f"frame.{isymbol}")
        os.makedirs(out, exist_ok=True)
        
        atom_numbers_list = []
        coords_list = [] 

        
        with h5py.File(os.path.join(out, "DM.h5"), 'w') as fid:
            icount = 0
            for iID in iID_lists:
                file  = f"{input_path}/{iID}.chk"

                h5dat = h5py.File(file, 'r')
                atom_numbers = h5dat['atomic_numbers'][:]
                coords = h5dat['coords'][:] *  bohr_to_angstrom


                atom_numbers_list.append(atom_numbers)
                coords_list.append(coords)

                nsites = len(atom_numbers)
                assert nsites == len(coords)

                if get_DM:
                    dm = h5dat['dm'][:]
                    R_cur = [0,0,0]
                    matrix_dict = dict()

                    for index_site_i in range(nsites):
                        for index_site_j in range(nsites):
                            key_str = f"{index_site_i + 1}_{index_site_j + 1}_{R_cur[0]}_{R_cur[1]}_{R_cur[2]}"

                            norb_i = site_norbits_dict[atomic_num_dict_r[atom_numbers[index_site_i]]]
                            norb_j = site_norbits_dict[atomic_num_dict_r[atom_numbers[index_site_j]]]

                            ist = int(np.sum(np.array([site_norbits_dict[atomic_num_dict_r[atom_numbers[_ii]]] for _ii in range(index_site_i)])))
                            jst = int(np.sum(np.array([site_norbits_dict[atomic_num_dict_r[atom_numbers[_jj]]] for _jj in range(index_site_j)])))

                            mat = dm[ist:ist+norb_i, jst:jst+norb_j]
                            if abs(mat).max() < 1e-10:
                                continue
                            
                            mat = U_orbital.transform(mat, orbital_types_dict[atomic_num_dict_r[atom_numbers[index_site_i]]], 
                                                        orbital_types_dict[atomic_num_dict_r[atom_numbers[index_site_j]]])

                            matrix_dict[key_str] = mat   
                    
                    icount+=1
                    default_group = fid.create_group(str(icount)) 
                    for key_str, value in matrix_dict.items():
                        default_group[key_str] = value
                
                else:
                    raise NotImplementedError("Only support get_DM=True.")
                
            coords_list = np.concatenate(coords_list, axis=0)
            atom_numbers_list = np.concatenate(atom_numbers_list, axis=0)

            np.savetxt(os.path.join(out, "positions.dat"), coords_list)
            np.savetxt(os.path.join(out, "atomic_numbers.dat"), atom_numbers_list, fmt='%d')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="./")
    parser.add_argument("-o", "--output_path", type=str, default="./")
    parser.add_argument("-s", "--data_name", type=str, default="C7H10O2")
    parser.add_argument("--get_DM", type=bool, default=True)
    args = parser.parse_args()
    
    _pyscf_parse(input_path = args.input_path,
                  output_path = args.output_path,
                  data_name = args.data_name,
                  get_DM = args.get_DM)

if __name__ == "__main__":
    main()
