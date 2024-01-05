from typing import Dict, Any, List, Callable, Union, Optional
import os
import glob

import numpy as np
import h5py
from ase import Atoms
from ase.io import Trajectory

import torch

from .. import (
    AtomicData,
    AtomicDataDict,
)
from ..transforms import TypeMapper, OrbitalMapper
from ._base_datasets import AtomicDataset, AtomicInMemoryDataset
#from dptb.nn.hamiltonian import E3Hamiltonian
from dptb.data.interfaces.ham_to_feature import ham_block_to_feature
from dptb.utils.tools import j_loader

class _TrajData(object):
    '''
    Input files format in a trajectory (shape):
    "info.json": optional, includes infomation in the data files.
                 can be provided in the base (upper level) folder, or assign in each trajectory.
    "cell.dat": fixed cell (3, 3) or variable cells (nframes, 3, 3). Unit: Angstrom
    "atomic_numbers.dat": (natoms) or (nframes, natoms)
    "positions.dat": concentrate all positions in one file, (nframes * natoms, 3). Can be cart or frac.

    Optional data files:
    "eigenvalues.npy": concentrate all engenvalues in one file, (nframes, nkpoints, nbands)
    "kpoints.npy": MUST be provided when loading `eigenvalues.npy`, (nkpoints, 3) or (nframes, nkpints, 3)
    "hamiltonians.h5": h5 file storing atom-wise hamiltonian blocks labeled by frames id and `i0_jR_Rx_Ry_Rz`.
    "overlaps.h5": the same format of overlap blocks as `hamiltonians.h5`
    '''
    
    def __init__(self, 
                 root: str, 
                 AtomicData_options: Dict[str, Any] = {},
                 get_Hamiltonian = False,
                 get_eigenvalues = False,
                 info = None,
                 _clear = False):
        self.root = root
        self.AtomicData_options = AtomicData_options
        self.info = info

        self.data = {}
        # load cell
        cell = np.loadtxt(os.path.join(root, "cell.dat"))
        if cell.shape[0] == 3:
            # same cell size, then copy it to all frames.
            cell = np.expand_dims(cell, axis=0)
            self.data["cell"] = np.broadcast_to(cell, (self.info["nframes"], 3, 3))
        elif cell.shape[0] == self.info["nframes"] * 3:
            self.data["cell"] = cell.reshape(self.info["nframes"], 3, 3)
        else:
            raise ValueError("Wrong cell dimensions.")
        
        # load atomic numbers
        atomic_numbers = np.loadtxt(os.path.join(root, "atomic_numbers.dat"))
        if atomic_numbers.shape[0] == self.info["natoms"]:
            # same atomic_numbers, copy it to all frames.
            atomic_numbers = np.expand_dims(atomic_numbers, axis=0)
            self.data["atomic_numbers"] = np.broadcast_to(atomic_numbers, (self.info["nframes"], 
                                                                           self.info["natoms"]))
        elif atomic_numbers.shape[0] == self.info["natoms"] * self.info["nframes"]:
            self.data["atomic_numbers"] = atomic_numbers.reshape(self.info["nframes"],
                                                                 self.info["natoms"])
        else:
            raise ValueError("Wrong atomic_number dimensions.")
        
        # load positions, stored as cartesion no matter what provided.
        pos = np.loadtxt(os.path.join(root, "positions.dat"))
        assert pos.shape[0] == self.info["nframes"] * self.info["natoms"]
        pos = pos.reshape(self.info["nframes"], self.info["natoms"], 3)
        # ase use cartesian by default.
        if self.info["pos_type"] == "cart" or self.info["pos_type"] == "ase":
            self.data["pos"] = pos
        elif self.info["pos_type"] == "frac":
            self.data["pos"] = pos @ self.data["cell"]
        else:
            raise NameError("Position type must be cart / frac.")
        
        # load optional data files
        if os.path.exists(os.path.join(self.root, "eigenvalues.npy")) and get_eigenvalues==True:
            assert "bandinfo" in self.info, "`bandinfo` must be provided in `info.json` for loading eigenvalues."
            assert os.path.exists(os.path.join(self.root, "kpoints.npy"))
            kpoints = np.load(os.path.join(self.root, "kpoints.npy"))
            if kpoints.ndim == 2:
                # same kpoints, then copy it to all frames.
                if kpoints.shape[0] == self.info["bandinfo"]["nkpoints"]:
                    kpoints = np.expand_dims(kpoints, axis=0)
                    self.data["kpoints"] = np.broadcast_to(kpoints, (self.info["nframes"], 
                                                                     self.info["bandinfo"]["nkpoints"], 3))
                else:
                    raise ValueError("kpoints in `.npy` file not equal to nkpoints in bandinfo. ")
            elif atomic_numbers.shape[0] == self.info["nframes"]:
                self.data["kpoints"] = kpoints
            else:
                raise ValueError("Wrong kpoint dimensions.")
            eigenvalues = np.load(os.path.join(self.root, "eigenvalues.npy"))
            # special case: trajectory contains only one frame
            if eigenvalues.ndim == 2:
                eigenvalues = np.expand_dims(eigenvalues, axis=0)
            assert eigenvalues.shape[0] == self.info["nframes"]
            assert eigenvalues.shape[1] == self.info["bandinfo"]["nkpoints"]
            assert eigenvalues.shape[2] == self.info["bandinfo"]["nbands"]
            self.data["eigenvalues"] = eigenvalues
            #self.data["eigenvalues"] = eigenvalues.reshape(self.info["nframes"], 
            #                                               self.info["bandinfo"]["nkpoints"], 
            #                                               self.info["bandinfo"]["nbands"])            
        if os.path.exists(os.path.join(self.root, "hamiltonians.h5")) and get_Hamiltonian==True:
            self.data["hamiltonian_blocks"] = h5py.File(os.path.join(self.root, "hamiltonians.h5"), "r")
            if os.path.exists(os.path.join(self.root, "overlaps.h5")):
                self.data["overlap_blocks"] = h5py.File(os.path.join(self.root, "overlaps.h5"), "r")
        
        # this is used to clear the tmp files to load ase trajectory only.
        if _clear:
            os.remove(os.path.join(root, "positions.dat"))
            os.remove(os.path.join(root, "cell.dat"))
            os.remove(os.path.join(root, "atomic_numbers.dat"))

    @classmethod
    def from_ase_traj(cls,
                      root: str, 
                      AtomicData_options: Dict[str, Any] = {},
                      get_Hamiltonian = False,
                      get_eigenvalues = False,
                      info = None):

        traj_file = glob.glob(f"{root}/*.traj")
        assert len(traj_file) == 1, print("only one ase trajectory file can be provided.")
        traj = Trajectory(traj_file[0], 'r')
        positions = []
        cell = []
        atomic_numbers = []
        for atoms in traj:
            positions.append(atoms.get_positions())
            cell.append(atoms.get_cell())
            atomic_numbers.append(atoms.get_atomic_numbers())
        positions = np.array(positions)
        positions = positions.reshape(-1, 3)
        cell = np.array(cell)
        cell = cell.reshape(-1, 3)
        atomic_numbers = np.array(atomic_numbers)
        atomic_numbers = atomic_numbers.reshape(-1, 1)
        np.savetxt(os.path.join(root, "positions.dat"), positions)
        np.savetxt(os.path.join(root, "cell.dat"), cell)
        np.savetxt(os.path.join(root, "atomic_numbers.dat"), atomic_numbers, fmt='%d')

        return cls(root=root,
                   AtomicData_options=AtomicData_options,
                   get_Hamiltonian=get_Hamiltonian,
                   get_eigenvalues=get_eigenvalues,
                   info=info,
                   _clear=True)
        
    def toAtomicDataList(self, idp: TypeMapper = None):
        data_list = []
        for frame in range(self.info["nframes"]):
            atomic_data = AtomicData.from_points(
                pos = self.data["pos"][frame][:],
                cell = self.data["cell"][frame][:],
                atomic_numbers = self.data["atomic_numbers"][frame],
                # pbc is stored in AtomicData_options now.
                #pbc = self.info["pbc"], 
                **self.AtomicData_options)
            if "hamiltonian_blocks" in self.data:
                assert idp is not None, "LCAO Basis must be provided  in `common_option` for loading Hamiltonian."
                if "overlap_blocks" not in self.data:
                    self.data["overlap_blocks"] = False
                # e3 = E3Hamiltonian(idp=idp, decompose=True)
                ham_block_to_feature(atomic_data, idp, 
                                     self.data["hamiltonian_blocks"][str(frame+1)], 
                                     self.data["overlap_blocks"][str(frame+1)])
                # with torch.no_grad():
                #     atomic_data = e3(atomic_data.to_dict())
                # atomic_data = AtomicData.from_dict(atomic_data)
            if "eigenvalues" in self.data and "kpoints" in self.data:
                assert "bandinfo" in self.info, "`bandinfo` must be provided in `info.json` for loading eigenvalues."
                bandinfo = self.info["bandinfo"]
                atomic_data[AtomicDataDict.KPOINT_KEY] = torch.as_tensor(self.data["kpoints"][frame][:], 
                                                                         dtype=torch.get_default_dtype())
                if bandinfo["emin"] is not None and bandinfo["emax"] is not None:
                    atomic_data[AtomicDataDict.ENERGY_WINDOWS_KEY] = torch.as_tensor([bandinfo["emin"], bandinfo["emax"]], 
                                                                                     dtype=torch.get_default_dtype())
                if bandinfo["band_min"] is not None and bandinfo["band_max"] is not None:
                    atomic_data[AtomicDataDict.BAND_WINDOW_KEY] = torch.as_tensor([bandinfo["band_min"], bandinfo["band_max"]], 
                                                                                  dtype=torch.get_default_dtype())
                    atomic_data[AtomicDataDict.ENERGY_EIGENVALUE_KEY] = torch.as_tensor(self.data["eigenvalues"][frame][bandinfo["band_min"]:bandinfo["band_max"]], 
                                                                                dtype=torch.get_default_dtype())
                else:
                    atomic_data[AtomicDataDict.ENERGY_EIGENVALUE_KEY] = torch.as_tensor(self.data["eigenvalues"][frame], 
                                                                                dtype=torch.get_default_dtype())
            data_list.append(atomic_data)
        return data_list
        

class DefaultDataset(AtomicInMemoryDataset):

    def __init__(
            self,
            root: str,
            info_files: Dict[str, Dict],
            url: Optional[str] = None,                    # seems useless but can't be remove
            include_frames: Optional[List[int]] = None,   # maybe support in future
            type_mapper: TypeMapper = None,
            get_Hamiltonian: bool = False,
            get_eigenvalues: bool = False,
    ):
        self.root = root
        self.url = url
        self.info_files = info_files
        # The following flags are stored to label dataset.
        self.get_Hamiltonian = get_Hamiltonian
        self.get_eigenvalues = get_eigenvalues

        # load all data files            
        self.raw_data = []
        for file in self.info_files.keys():
            # get the info here
            info = info_files[file]
            assert "AtomicData_options" in info
            AtomicData_options = info["AtomicData_options"]
            assert "r_max" in AtomicData_options
            assert "pbc" in AtomicData_options
            if info["pos_type"] == "ase":
                subdata = _TrajData.from_ase_traj(os.path.join(self.root, file), 
                                AtomicData_options,
                                get_Hamiltonian, 
                                get_eigenvalues,
                                info=info)
            else:
                subdata = _TrajData(os.path.join(self.root, file), 
                                AtomicData_options,
                                get_Hamiltonian, 
                                get_eigenvalues,
                                info=info)
            self.raw_data.append(subdata)
        
        # The AtomicData_options is never used here.
        # Because we always return a list of AtomicData object in `get_data()`.
        # That is, AtomicInMemoryDataset will not use AtomicData_options to build any AtomicData here.
        super().__init__(
            file_name=None, # this seems not important too.
            url=url,
            root=root,
            AtomicData_options={},  # we do not pass anything here.
            include_frames=include_frames,
            type_mapper=type_mapper,
        )

    def get_data(self):
        all_data = []
        for subdata in self.raw_data:
            # the type_mapper here is loaded in PyG `dataset` type as `transform` attritube
            # so the OrbitalMapper can be accessed by self.transform here
            subdata_list = subdata.toAtomicDataList(self.transform)
            all_data += subdata_list
        return all_data
    
    @property
    def raw_file_names(self):
        # TODO: this is not implemented.
        return "Null"

    @property
    def raw_dir(self):
        # TODO: this is not implemented.
        return self.root
    