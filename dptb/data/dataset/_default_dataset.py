from typing import Dict, Any, List, Callable, Union, Optional
import os

import numpy as np
import h5py

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
    Input file format in a trajectory (shape):
    "info.json": includes infomation in the data files.
    "cell.dat": fixed cell (3, 3) or variable cells (nframes, 3, 3). Unit: Angstrom
    "atomic_numbers.dat": (natoms) or (nframes, natoms)
    "positions.dat": concentrate all positions in one file, (nframes * natoms, 3). Can be cart or frac.

    Optional:
    "eigenvalues.npy": concentrate all engenvalues in one file, (nframes, nkpoints, nbands)
    "kpoints.npy": MUST be provided when loading `eigenvalues.npy`, (nkpoints, 3) or (nframes, nkpints, 3)
    "hamiltonians.h5": h5 file storing atom-wise hamiltonian blocks labeled by frames id and `i0_jR_Rx_Ry_Rz`.
    "overlaps.h5": the same format of overlap blocks as `hamiltonians.h5`
    '''
    
    def __init__(self, root: str, AtomicData_options: Dict[str, Any] = {},):
        self.root = root
        self.AtomicData_options = AtomicData_options
        self.info = j_loader(os.path.join(root, "info.json"))

        self.data = {}
        cell = np.loadtxt(os.path.join(root, "cell.dat"))
        if cell.shape[0] == 3:
            # same cell size, then copy it to all frames.
            cell = np.expand_dims(cell, axis=0)
            self.data["cell"] = np.broadcast_to(cell, (self.info["nframes"], 3, 3))
        elif cell.shape[0] == self.info["nframes"] * 3:
            self.data["cell"] = cell.reshape(self.info["nframes"], 3, 3)
        else:
            raise ValueError("Wrong cell dimensions.")
        atomic_numbers = np.loadtxt(os.path.join(root, "atomic_numbers.dat"))
        if len(atomic_numbers.shape) == 1:
            # same atomic_numbers, copy it to all frames.
            if atomic_numbers.shape[0] == self.info["natoms"]:
                atomic_numbers = np.expand_dims(atomic_numbers, axis=0)
                self.data["atomic_numbers"] = np.broadcast_to(atomic_numbers, (self.info["nframes"], 
                                                                               self.info["natoms"]))
            else:
                raise ValueError("Atomic numbers not equal to natoms in info.json. ")
        elif atomic_numbers.shape[0] == self.info["natoms"] * self.info["nframes"]:
            self.data["atomic_numbers"] = atomic_numbers.reshape(self.info["nframes"],
                                                                 self.info["natoms"])
        else:
            raise ValueError("Wrong atomic_number dimensions.")
        pos = np.loadtxt(os.path.join(root, "positions.dat"))
        assert pos.shape[0] == self.info["nframes"] * self.info["natoms"]
        pos = pos.reshape(self.info["nframes"], self.info["natoms"], 3)
        if self.info["pos_type"] == "cart":
            self.data["pos"] = pos
        elif self.info["pos_type"] == "frac":
            self.data["pos"] = pos @ self.data["cell"]
        else:
            raise NameError("Position type must be cart / frac.")

        if os.path.exists(os.path.join(self.root, "eigenvalues.npy")):
            assert os.path.exists(os.path.join(self.root, "kpoints.npy"))
            kpoints = np.load(os.path.join(self.root, "kpoints.npy"))
            if len(kpoints.shape) == 2:
                # same kpoints, then copy it to all frames.
                if kpoints.shape[0] == self.info["bandinfo"]["nkpoints"]:
                    kpoints = np.expand_dims(kpoints, axis=0)
                    self.data["kpoints"] = np.broadcast_to(kpoints, (self.info["nframes"], 
                                                                     self.info["bandinfo"]["nkpoints"], 3))
                else:
                    raise ValueError("kpoints in .npy not equal to nkpoints in bandinfo. ")
            elif atomic_numbers.shape[0] == self.info["nframes"]:
                self.data["kpoints"] = kpoints
            else:
                raise ValueError("Wrong kpoint dimensions.")
            eigenvalues = np.load(os.path.join(self.root, "eigenvalues.npy"))
            assert eigenvalues.shape[0] == self.info["nframes"]
            assert eigenvalues.shape[1] == self.info["bandinfo"]["nkpoints"]
            assert eigenvalues.shape[2] == self.info["bandinfo"]["nbands"]
            self.data["eigenvalues"] = eigenvalues
            #self.data["eigenvalues"] = eigenvalues.reshape(self.info["nframes"], 
            #                                               self.info["bandinfo"]["nkpoints"], 
            #                                               self.info["bandinfo"]["nbands"])            
        if os.path.exists(os.path.join(self.root, "hamiltonians.h5")):
            self.data["hamiltonian_blocks"] = h5py.File(os.path.join(self.root, "hamiltonians.h5"), "r")
        if os.path.exists(os.path.join(self.root, "overlaps.h5")):
            self.data["overlap_blocks"] = h5py.File(os.path.join(self.root, "overlaps.h5"), "r")
        
    def toAtomicDataList(self, idp: TypeMapper = None):
        data_list = []
        for frame in range(self.info["nframes"]):
            atomic_data = AtomicData.from_points(
                pos = self.data["pos"][frame][:],
                cell = self.data["cell"][frame][:],
                atomic_numbers = self.data["atomic_numbers"][frame],
                pbc = self.info["pbc"],
                **self.AtomicData_options)
            if "hamiltonian_blocks" in self.data:
                assert idp is not None, "LCAO Basis must be provided for loading Hamiltonian."
                if "overlap_blocks" not in self.data:
                    self.data["overlap_blocks"] = False
                # e3 = E3Hamiltonian(idp=idp, decompose=True)
                ham_block_to_feature(atomic_data, idp, 
                                     self.data["hamiltonian_blocks"][str(frame)], 
                                     self.data["overlap_blocks"][str(frame)])
                # with torch.no_grad():
                #     atomic_data = e3(atomic_data.to_dict())
                # atomic_data = AtomicData.from_dict(atomic_data)
            if "eigenvalues" in self.data and "kpoints" in self.data:
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
            prefix: Optional[str] = None,
            url: Optional[str] = None,
            AtomicData_options: Dict[str, Any] = {},
            include_frames: Optional[List[int]] = None,
            type_mapper: TypeMapper = None,
    ):
        self.file_name = []
        for dir_name in os.listdir(root):
            if os.path.isdir(os.path.join(root, dir_name)):
                if prefix is not None:
                    if dir_name[:len(prefix)] == prefix:
                        self.file_name.append(dir_name)
                else:
                    self.file_name.append(dir_name)
        # the type_mapper must be stored here in order to load Hamiltonian.
        #all_basis = []
        #for file in self.file_name:
        #    file_info = j_loader(os.path.join(file, "info.json"))
        #    all_basis.append(file_info["basis"])
        #sort_basis = {}
        #for basis in all_basis:
        #    for symbol, orbitals in basis.items():
        #        if symbol not in sort_basis:
        #            sort_basis[symbol] = orbitals
        #type_mapper = OrbitalMapper(sort_basis)
        super().__init__(
            file_name=self.file_name,
            url=url,
            root=root,
            AtomicData_options=AtomicData_options,
            include_frames=include_frames,
            type_mapper=type_mapper,
        )

    def setup_data(self):
        self.data = []
        for file in self.file_name:
            subdata = _TrajData(os.path.join(self.root, file), self.AtomicData_options)
            self.data.append(subdata)

    def get_data(self):
        self.setup_data()
        all_data = []
        for subdata in self.data:
            # the type_mapper here is loaded in `dataset` type as `transform` attritube
            subdata_list = subdata.toAtomicDataList(self.transform)
            all_data += subdata_list
        return all_data
    
    @property
    def raw_file_names(self):
        return "Null"

    @property
    def raw_dir(self):
        return self.root
    