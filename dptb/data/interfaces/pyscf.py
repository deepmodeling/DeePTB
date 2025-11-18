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
import logging
import pickle
from dptb.utils.loggers import set_log_handles
from pathlib import Path


log=logging.getLogger(__name__)

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

def split_ids(symbol2ID, ratio_str:str="8:1:1"):
    ratios = [float(part) if part else 0 for part in ratio_str.split(":")]
    total = sum(ratios)
    assert total > 0
    ratios = [float(ratio) / total for ratio in ratios]
    assert len(ratios) == 3
    
    log_path=None
    set_log_handles(logging.INFO, Path(log_path) if log_path else None) 

    log.info(f"Splitting data with ratios: {ratios}")


    chem_symbols = list(symbol2ID.keys())
    idlist = []

    for isym in chem_symbols:
        idlist.extend(symbol2ID[isym])
    rds = np.random.RandomState(1)
    rand_keys = rds.choice(idlist, len(idlist), replace=False)
    nframes = len(rand_keys)
    ntrain = int(nframes * ratios[0])
    nval = int(nframes * ratios[1])
    ntest = nframes - ntrain - nval

    log.info(f"Splitting data with ntrain: {ntrain}, nval: {nval}, ntest: {ntest}")

    train_set = rand_keys[:ntrain]
    val_set = rand_keys[ntrain:ntrain + nval]
    test_set = rand_keys[ntrain + nval:]

    train_sym_ids = {}
    val_sym_ids = {}
    test_sym_ids = {}

    for isym in chem_symbols:
        for iID in symbol2ID[isym]:
            if iID in train_set:
                if isym not in train_sym_ids:
                    train_sym_ids[isym] = []
                train_sym_ids[isym].append(iID)
            
            elif iID in val_set:
                if isym not in val_sym_ids:
                    val_sym_ids[isym] = []
                val_sym_ids[isym].append(iID)
            
            elif iID in test_set:
                if isym not in test_sym_ids:
                    test_sym_ids[isym] = []
                test_sym_ids[isym].append(iID)
            
            else:
                raise ValueError("Invalid ID.")
            
    return train_sym_ids, val_sym_ids, test_sym_ids

def _pyscf_parse_qm9_split(input_path,
                     output_path, 
                     split_ratio=None,
                     get_DM=True):
    
    assert split_ratio is not None
    train_sym_ids, val_sym_ids, test_sym_ids = split_ids(symbol2ID=symbol2ID, ratio_str=split_ratio)
    
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

    
    for setname in ["train", "val", "test"]:
        
        if setname == "train":
            data_name = train_sym_ids 
        elif setname == "val":
            data_name = val_sym_ids
        else:
            data_name = test_sym_ids

        out2 = os.path.join(output_path, setname)

        for isymbol, iID_lists in data_name.items():
            out = os.path.join(out2, f"frame.{isymbol}")
            os.makedirs(out, exist_ok=True)

            # atom_numbers_list = []
            # coords_list = [] 

            struct = {}
            with h5py.File(os.path.join(out, "DM.h5"), 'w') as fid, \
                    open(os.path.join(out, "structure.pkl"), 'wb') as pid:
                
                icount = 0
                for iID in iID_lists:
                    file  = f"{input_path}/{iID}.chk"
                    try:
                        coords, atom_numbers, matrix_dict = _chkfile_parse(file, 
                                                                           site_norbits_dict, 
                                                                           orbital_types_dict,
                                                                           get_DM)    
                    except:    
                        log.info(f"Error in {file}, skip.")
                        continue
                        # atom_numbers_list.append(atom_numbers)
                        # coords_list.append(coords) 
                    icount+=1
                    default_group = fid.create_group(str(icount)) 
                    struct[str(icount)] = {}

                    for key_str, value in matrix_dict.items():
                        default_group[key_str] = value

                    struct[str(icount)]["positions"] = coords
                    struct[str(icount)]["atomic_numbers"] = atom_numbers
                    
  
                pickle.dump(struct, pid)



def _pyscf_parse_qm9(input_path, 
                     output_path, 
                     data_name=None, 
                     get_DM=True):
    
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)

    chem_symbols = list(symbol2ID.keys())
    
    is_collect = True
    if data_name is None:
        is_collect = False
        # no symbol is provided, will process each file and will not collect them.
        log.info("No symbol is provided, will process each file and will not collect them.")
        data_name =  glob.glob(f"{input_path}/*.chk")

    elif isinstance(data_name, str):
        if data_name.lower() == 'all':
            data_name = chem_symbols
        else:
            assert data_name in chem_symbols
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

    if is_collect:
        for isymbol  in data_name:
            iID_lists  = symbol2ID[isymbol]
            out = os.path.join(output_path, f"frame.{isymbol}")
            os.makedirs(out, exist_ok=True)

            # atom_numbers_list = []
            # coords_list = [] 

            struct = {}
            with h5py.File(os.path.join(out, "DM.h5"), 'w') as fid, \
                    open(os.path.join(out, "structure.pkl"), 'wb') as pid:
                
                icount = 0
                for iID in iID_lists:
                    file  = f"{input_path}/{iID}.chk"
                    coords, atom_numbers, matrix_dict = _chkfile_parse(file, 
                                                                       site_norbits_dict, 
                                                                       orbital_types_dict,
                                                                       get_DM)    
                    # atom_numbers_list.append(atom_numbers)
                    # coords_list.append(coords) 
                    icount+=1
                    default_group = fid.create_group(str(icount)) 
                    struct[str(icount)] = {}

                    for key_str, value in matrix_dict.items():
                        default_group[key_str] = value

                    struct[str(icount)]["positions"] = coords
                    struct[str(icount)]["atomic_numbers"] = atom_numbers
                
                pickle.dump(struct, pid)
            
            # coords_list = np.concatenate(coords_list, axis=0)
            # atom_numbers_list = np.concatenate(atom_numbers_list, axis=0)
            # np.savetxt(os.path.join(out, "positions.dat"), coords_list)
            # np.savetxt(os.path.join(out, "atomic_numbers.dat"), atom_numbers_list, fmt='%d')
    else:
        for idat in data_name:
            file_name = idat.split('/')[-1]
            file_id = file_name.split('.')[0]
            out = os.path.join(output_path, f"frame.{file_id}")
            os.makedirs(out, exist_ok=True)
            
            struct = {}
            with h5py.File(os.path.join(out, "DM.h5"), 'w') as fid, \
                open(os.path.join(out, "structure.pkl"), 'wb') as pid:
                
                coords, atom_numbers, matrix_dict = _chkfile_parse(idat, 
                                                                   site_norbits_dict, 
                                                                   orbital_types_dict,
                                                                   get_DM)    

                # np.savetxt(os.path.join(out, "positions.dat"), coords)
                # np.savetxt(os.path.join(out, "atomic_numbers.dat"), atom_numbers, fmt='%d')
                default_group = fid.create_group("1") 
                struct['1'] = {}

                for key_str, value in matrix_dict.items():
                    default_group[key_str] = value

                struct['1']["positions"] = coords
                struct['1']["atomic_numbers"] = atom_numbers

                pickle.dump(struct, pid)

def split(input_path, output_path, ratio_str:str="8:1:1"):
    # ratio str a:b:c
    ratios = [float(part) if part else 0 for part in ratio_str.split(":")]
    total = sum(ratios)
    assert total > 0
    ratios = [float(ratio) / total for ratio in ratios]

    assert os.path.exists(os.path.join(input_path, "structure.pkl")) and os.path.exists(os.path.join(input_path, "DM.h5"))
    
    os.makedirs(output_path, exist_ok=True)
    log_path = os.path.join(str(output_path), "log.txt")

    set_log_handles(logging.INFO, Path(log_path) if log_path else None) 

    log.info(f"Splitting data with ratios: {ratios}")

    with open(os.path.join(input_path, "structure.pkl"), 'rb') as pid:
        struct = pickle.load(pid)

    nframes = len(struct)
    ntrain = int(nframes * ratios[0])
    nval = int(nframes * ratios[1])
    ntest = nframes - ntrain - nval

    log.info(f"Splitting data with ntrain: {ntrain}, nval: {nval}, ntest: {ntest}")
    
    # setup seed
    rds = np.random.RandomState(1)
    rand_keys = rds.choice(list(struct.keys()), nframes, replace=False)
    
    assert ntrain > 0
    train_set = {key: struct[key] for key in rand_keys[:ntrain]}
    os.makedirs(os.path.join(output_path, "train.0"), exist_ok=True)
    with open(os.path.join(output_path, "train.0", "structure.pkl"), 'wb') as pid:
        pickle.dump(train_set, pid)
    
    if nval > 0:
        val_set = {key: struct[key] for key in rand_keys[ntrain:ntrain + nval]}
        os.makedirs(os.path.join(output_path, "val.0"), exist_ok=True)
        with open(os.path.join(output_path, "val.0", "structure.pkl"), 'wb') as pid:
            pickle.dump(val_set, pid)

    if ntest > 0:
        test_set = {key: struct[key] for key in rand_keys[ntrain + nval:]}
        os.makedirs(os.path.join(output_path, "test.0"), exist_ok=True)
        with open(os.path.join(output_path, "test.0", "structure.pkl"), 'wb') as pid:
            pickle.dump(test_set, pid)

    DM = h5py.File(os.path.join(input_path, "DM.h5"), 'r')

    with h5py.File(os.path.join(output_path, "train.0", "DM.h5"), 'w') as fid:
        for key in train_set.keys():
            fid.create_group(key)
            for key_str, value in DM[key].items():
                    fid[key][key_str] = value[:]

    if nval >0:
        with h5py.File(os.path.join(output_path, "val.0", "DM.h5"), 'w') as fid:
            for key in val_set.keys():
                fid.create_group(key)
                for key_str, value in DM[key].items():
                    fid[key][key_str] = value[:]
    
    if ntest >0:
        with h5py.File(os.path.join(output_path, "test.0", "DM.h5"), 'w') as fid:
            for key in test_set.keys():
                fid.create_group(key)
                for key_str, value in DM[key].items():
                    fid[key][key_str] = value[:]


def main():
    parser = argparse.ArgumentParser(
                description="DeepTB parse QM9 data sets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")
    parser_collect = subparsers.add_parser(
        "collect",
        help="collect data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_collect.add_argument("-i", "--input_path", type=str, default="./")
    parser_collect.add_argument("-o", "--output_path", type=str, default="./")
    parser_collect.add_argument("-s", "--data_name", type=str, default=None)
    parser_collect.add_argument("--get_DM", type=bool, default=True)

    parser_split = subparsers.add_parser(
        "split",
        help="split data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_split.add_argument("-i", "--input_path", type=str, default="./")
    parser_split.add_argument("-o", "--output_path", type=str, default="./")
    parser_split.add_argument("-r", "--ratio_str", type=str, default="8:1:1")

    parser_ctsp = subparsers.add_parser(
        "ctsp",
        help="collect and split data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_ctsp.add_argument("-i", "--input_path", type=str, default="./")
    parser_ctsp.add_argument("-o", "--output_path", type=str, default="./")
    parser_ctsp.add_argument("-r", "--ratio_str", type=str, default="8:1:1")
    parser_ctsp.add_argument("--get_DM", type=bool, default=True)

    args = parser.parse_args()
    if args.command == "collect":
        _pyscf_parse_qm9(input_path = args.input_path,
                     output_path = args.output_path,
                     data_name = args.data_name,
                     get_DM = args.get_DM)


    elif args.command == "split":
        split(input_path = args.input_path,
                     output_path = args.output_path,
                     ratio_str = args.ratio_str)
    
    elif args.command == "ctsp":
        _pyscf_parse_qm9_split(input_path = args.input_path,
                        output_path = args.output_path,
                        split_ratio = args.ratio_str,
                        get_DM = args.get_DM)
    
    else:
        raise ValueError("Invalid command.")
    
    
if __name__ == "__main__":
    main()
