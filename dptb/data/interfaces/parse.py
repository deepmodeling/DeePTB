import os
import glob

from abc import ABC, abstractmethod

import numpy as np
from pathlib import Path
import h5py
import ase
import pickle
import lmdb
import yaml
import hashlib
import dpdata
from .. import _keys
from dptb.utils.tools import j_must_have

def find_target_line(f, target):
    line = f.readline()
    while line:
        if target in line:
            return line
        line = f.readline()
    return None

class Parser(ABC):
    def __init__(
            self,
            root,
            prefix,
            ):
        """All DFT parser need to inherit this class and implement the abstract methods.
        The subclass can call the check_structure, check_eigenvalue, check_blocks to check the parsed data structure.
        To see if the parsed data structure are in the right shape and dtype.

        Parameters
        ----------
        root : The root of the DFT calculation output files
        prefix : str
            The prefix of the DFT calculation folders or files
        """

        self.root = root
        self.predix = prefix

        if isinstance(root, list) and all(isinstance(item, str) for item in root):
            self.raw_datas = root
        else:
            self.raw_datas = glob.glob(root + '/*' + prefix + '*')
    
    def __len__(self):
        return len(self.raw_datas)
    
    def __getitem__(self, idx):
        return self.raw_datas[idx]
    
    @abstractmethod
    def get_structure(self, idx):
        pass # return a dict of structure information, keys includes: [_keys.ATOMIC_NUMBERS_KEY, _keys.POSITIONS_KEY, _keys.CELL_KEY, _keys.PBC_KEY], CELL could be None if no PBC is applied
    
    # def get_hash(self, idx):

    #     buffer = yaml.dump(self.get_structure(idx=idx)).encode("ascii")
    #     # And hash it:
    #     param_hash = hashlib.sha1(buffer).hexdigest()
    #     return param_hash
    
    def formula(self, idx):
        structure = self.get_structure(idx)
        atomic_number = j_must_have(structure, _keys.ATOMIC_NUMBERS_KEY)

        return ase.Atoms(numbers=atomic_number).get_chemical_formula()

    @abstractmethod
    def get_eigenvalue(self, idx):
        pass # return a dict with energy eigenvalue and kpoint as key, keys includes: [_keys.ENERGY_EIGENVALUE_KEY, _keys.KPOINT_KEY]

    @abstractmethod
    def get_blocks(self, idx, hamiltonian: bool=False, overlap: bool=False, density_matrix: bool=False):
        pass # return a list of hamiltonian, overlap, density_matrix dict, with i_j_Rx_Ry_Rz as key, and the block as value

    # @abstractmethod
    # def get_field():
    #     pass
    
    def check_structure(self, idx):
        structure = self.get_structure(idx)
        atomic_number = j_must_have(structure, _keys.ATOMIC_NUMBERS_KEY)
        pos = j_must_have(structure, _keys.POSITIONS_KEY)
        cell = j_must_have(structure, _keys.CELL_KEY)
        pbc = j_must_have(structure, _keys.PBC_KEY)

        # check the shape
        assert len(pos.shape) == 3 and pos.shape[2] == 3, "The shape of structure should be (n_frame, natom, 3)"
        n_frame, natom, _ = pos.shape
        assert len(cell.shape) == 3 and cell.shape[0] == n_frame and cell.shape[1] == 3 and cell.shape[2] == 3, "The shape of cell should be (n_frame, 3, 3)"
        assert len(atomic_number.shape) == 1 and atomic_number.shape[0] == natom, "The shape of atomic_number should be (natom,)"
        assert len(pbc.shape) == 1 and pbc.shape[0] == 3, "The shape of pbc should be (3)"
        
        # check dtype
        assert atomic_number.dtype == np.int32, "The dtype of atomic_number should be int"
        assert cell.dtype == np.float32, "The dtype of cell should be float"
        assert pos.dtype == np.float32, "The dtype of pos should be float"
        assert pbc.dtype == np.bool8, "The dtype of pbc should be bool"

        return True
    
    def check_eigenvalue(self, idx):
        eigstatus = self.get_eigenvalue(idx)
        eigs = j_must_have(eigstatus, _keys.ENERGY_EIGENVALUE_KEY)
        kpts = j_must_have(eigstatus, _keys.KPOINT_KEY)

        # check the shape
        assert len(eigs.shape) == 3, "The shape of eigenvalue should be (n_frame, n_kpt, n_band)"
        nf, nk, nb = eigs.shape
        assert len(kpts.shape) == 2 and kpts.shape[0] == nk and kpts.shape[1] == 3, "The shape of kpoint should be (n_kpt, 3)"

        # check dtype
        assert eigs.dtype == np.float32, "The dtype of eigenvalue should be float"
        assert kpts.dtype == np.float32, "The dtype of kpoint should be float"

        return True
    
    def check_blocks(self, idx, hamiltonian: bool=False, overlap: bool=False, density_matrix: bool=False):
        ham, ovp, dm = self.get_blocks(idx, hamiltonian, overlap, density_matrix)
        structure = self.get_structure(idx)
        if hamiltonian:
            assert ham is not None, "Hamiltonian should not be None"
            assert len(ham) == structure[_keys.POSITIONS_KEY].shape[0], "The number of hamiltonian blocks should be equal to the number of frames"

            assert all([isinstance(h, dict) for h in ham]), "Hamiltonian should be a list of dict"
            assert ham[0].get("0_0_0_0_0") is not None, "Hamiltonian should at least have key 0_0_0_0_0"
            assert ham[0].get("0_0_0_0_0").dtype in [np.float32, np.complex64], "The dtype of hamiltonian block should be float real or complex"

        if overlap:
            assert ovp is not None, "Overlap should not be None"
            assert len(ovp) == structure[_keys.POSITIONS_KEY].shape[0], "The number of overlap blocks should be equal to the number of frames"
        if density_matrix:
            assert dm is not None, "Density matrix should not be None"
            assert len(dm) == structure[_keys.POSITIONS_KEY].shape[0], "The number of density matrix blocks should be equal to the number of frames"

        return True

    def write_hdf5(self, idx, root, eigenvalue: bool=False, hamiltonian: bool=False, overlap: bool=False, density_matrix: bool=False):
        pass
    
    def write_dat(self, idx, root, eigenvalue=False, hamiltonian=False, overlap=False, density_matrix=False):
        # write structure
        os.makedirs(root, exist_ok=True)
       
        structure = self.get_structure(idx)

        out_dir = os.path.join(root, self.formula(idx=idx)+".{}".format(idx))
        os.makedirs(out_dir, exist_ok=True)
        # The abacus must have PBC, so here we save cell by default
        np.savetxt(os.path.join(out_dir, "cell.dat"), structure[_keys.CELL_KEY].reshape(-1, 3))
        np.savetxt(os.path.join(out_dir, "position.dat"), structure[_keys.POSITIONS_KEY].reshape(-1, 3))
        np.savetxt(os.path.join(out_dir, "atomic_number.dat"), structure[_keys.ATOMIC_NUMBERS_KEY], fmt='%d')
        np.savetxt(os.path.join(out_dir, "pbc.dat"), structure[_keys.PBC_KEY])


        # write eigenvalue
        if eigenvalue:
            eigstatus = self.get_eigenvalue(idx)
            np.save(os.path.join(out_dir, "kpoint.npy"), eigstatus[_keys.KPOINT_KEY])
            np.save(os.path.join(out_dir, "eigenvalue.npy"), eigstatus[_keys.ENERGY_EIGENVALUE_KEY].reshape(-1, eigstatus[_keys.ENERGY_EIGENVALUE_KEY].shape[-1]))

        # write blocks
        if any([hamiltonian is not None, overlap is not None, density_matrix is not None]):
            with open(os.path.join(out_dir, "basis.dat"), 'w') as f:
               f.write(str(self.get_basis(idx)))

        ham, ovp, dm = self.get_blocks(idx, hamiltonian, overlap, density_matrix)
        if hamiltonian:
            with h5py.File(os.path.join(out_dir, "hamiltonian.h5"), 'w') as fid:
                for i in range(len(ham)):
                    default_group = fid.create_group(str(i))
                    for key_str, value in ham[i].items():
                        default_group[key_str] = value
        
        if overlap:
            with h5py.File(os.path.join(out_dir, "overlap.h5"), 'w') as fid:
                for i in range(len(ovp)):
                    default_group = fid.create_group(str(i))
                    for key_str, value in ovp[i].items():
                        default_group[key_str] = value
        
        if density_matrix:
            with h5py.File(os.path.join(out_dir, "density_matrix.h5"), 'w') as fid:
                for i in range(len(dm)):
                    default_group = fid.create_group(str(i))
                    for key_str, value in dm[i].items():
                        default_group[key_str] = value

        return True
    
    def write_lmdb(self, idx, root, prefix, eigenvalue: bool=False, hamiltonian: bool=False, overlap: bool=False, density_matrix: bool=False):
        os.makedirs(root, exist_ok=True)
        out_dir = os.path.join(root, prefix+".lmdb")
        lmdb_env = lmdb.open(os.path.join(out_dir), map_size=1048576000000)
        structure = self.get_structure(idx)

        with lmdb_env.begin(write=True) as txn:
            if any([hamiltonian, overlap, density_matrix]):
                ham, ovp, dm = self.get_blocks(idx, hamiltonian, overlap, density_matrix)
            if eigenvalue:
                eigstatus = self.get_eigenvalue(idx)

            n_frames = structure[_keys.POSITIONS_KEY].shape[0]
            for nf in range(n_frames):
                data_dict = {}
                data_dict[_keys.ATOMIC_NUMBERS_KEY] = structure[_keys.ATOMIC_NUMBERS_KEY]
                data_dict[_keys.CELL_KEY] = structure[_keys.CELL_KEY][nf]
                data_dict[_keys.POSITIONS_KEY] = structure[_keys.POSITIONS_KEY][nf]
                data_dict[_keys.PBC_KEY] = structure[_keys.PBC_KEY]

                if eigenvalue:
                    data_dict[_keys.ENERGY_EIGENVALUE_KEY] = eigstatus[_keys.ENERGY_EIGENVALUE_KEY][nf]
                    data_dict[_keys.KPOINT_KEY] = eigstatus[_keys.KPOINT_KEY]

                if hamiltonian:
                    data_dict["hamiltonian"] = ham[nf]
                if overlap:
                    data_dict["overlap"] = ovp[nf]
                if density_matrix:
                    data_dict["density_matrix"] = dm[nf]

                
                data_dict = pickle.dumps(data_dict)
                entries = lmdb_env.stat()["entries"]
                txn.put(entries.to_bytes(length=4, byteorder='big'), data_dict)

        lmdb_env.close()
