import pytest
from dptb.data.dataset._default_dataset import DefaultDataset, _TrajData
from dptb.data.AtomicData import AtomicData
from dptb.data import AtomicDataDict
from dptb.data.transforms import OrbitalMapper
import os
import numpy as np
from pathlib import Path
from ase.io.trajectory import Trajectory
import torch as th
from dptb.tests.tstools import compare_tensors_as_sets

rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data/test_sktb/dataset")

class TestDefaultDatasetSKTB:
    rootdir = f"{rootdir}"
    basis = {"Si": ["3s", "3p"]}
    info_files = {'kpath_spk.0': {'nframes': 1,
      'natoms': 2,
      'pos_type': 'ase',
      'pbc': True,
      'r_max': 5.0,
       'er_max': 5.0,
       'oer_max': 2.5,
      'bandinfo': {'nkpoints': 61,
       'nbands': 14,
       'band_min': 0,
       'band_max': 6,
       'emin': -1.0,
       'emax': 10.0}}}
    idp = OrbitalMapper(basis)
    dataset = DefaultDataset(
        root=rootdir,
        type_mapper=idp,
        get_Hamiltonian=False,
        get_eigenvalues=True,
        get_overlap=False,
        get_DM=False,
        info_files = info_files)

    strase = Trajectory(f"{rootdir}/kpath_spk.0/xdat.traj",'r')
    kpoints = np.load(f"{rootdir}/kpath_spk.0/kpoints.npy")
    eigenvalues = np.load(f"{rootdir}/kpath_spk.0/eigenvalues.npy")

    def test_inparas(self):
        assert self.dataset.root == self.rootdir
        assert isinstance(self.dataset.type_mapper, OrbitalMapper)
        assert self.dataset.get_Hamiltonian == False
        assert self.dataset.get_eigenvalues == True
        assert self.dataset.get_overlap == False
        assert self.dataset.get_DM == False
        assert len(self.dataset.info_files) == 1
        assert self.dataset.info_files == self.info_files
        assert self.dataset.AtomicData_options == {}


    def test_raw_data(self):
        assert len(self.dataset.raw_data) == 1
        assert isinstance(self.dataset.raw_data[0], _TrajData)
        # assert self.dataset.raw_data[0].AtomicData_options == {'r_max': 5.0, 'er_max': 5.0, 'oer_max': 2.5, 'pbc': True}
        assert self.dataset.raw_data[0].info == self.info_files['kpath_spk.0']
        assert "bandinfo" in self.dataset.raw_data[0].info
        assert  list(self.dataset.raw_data[0].data.keys()) == (['cell', 'pos', 'atomic_numbers', 'kpoint', 'eigenvalue'])
        assert (np.abs(self.dataset.raw_data[0].data['cell'] - self.strase[0].cell) < 1e-6).all()
        assert (np.abs(self.dataset.raw_data[0].data['atomic_numbers'] - np.array([[14., 14.]])) < 1e-6).all()
        assert (np.abs(self.dataset.raw_data[0].data['pos'] - self.strase[0].positions) < 1e-6).all()
        assert self.dataset.raw_data[0].data['kpoint'].shape == (1, 61, 3)
        assert (np.abs(self.dataset.raw_data[0].data['kpoint'] - self.kpoints) < 1e-6).all()
        assert self.dataset.raw_data[0].data['eigenvalue'].shape == (1, 61, 14)
        assert (np.abs(self.dataset.raw_data[0].data['eigenvalue'] - self.eigenvalues) < 1e-6).all()

        assert "hamiltonian_blocks" not in self.dataset.raw_data[0].data
        assert "overlap_blocks" not in self.dataset.raw_data[0].data

    def test_get_data(self):
        expected_edge_index = th.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 1, 1, 1, 1, 1]])
        
        data = self.dataset.get_data()
        atomic_data = data[0]
        assert len(data) == 1  # 1 subdatasets
        assert isinstance(data, list)
        assert isinstance(data[0], AtomicData)

        assert (np.abs(atomic_data.pos.numpy() - self.strase[0].positions) < 1e-6).all()
        assert (np.abs(atomic_data.cell.numpy() - self.strase[0].cell) < 1e-6).all()

        assert compare_tensors_as_sets(atomic_data.edge_index.T, expected_edge_index.T) 
        # assert th.abs(atomic_data.edge_index - expected_edge_index).sum() < 1e-8
        assert atomic_data.node_features.shape == (2, 1)
        assert not "node_attrs" in data[0]
        assert not "batch" in data[0]

        assert atomic_data[AtomicDataDict.EDGE_FEATURES_KEY].shape == th.Size([56, 1])
        assert atomic_data[AtomicDataDict.NODE_FEATURES_KEY].shape == th.Size([2, 1])
        assert atomic_data[AtomicDataDict.EDGE_OVERLAP_KEY].shape == th.Size([56, 1])
        assert (np.abs(atomic_data[AtomicDataDict.KPOINT_KEY][0].numpy() 
                       - self.dataset.raw_data[0].data['kpoint'][0])<1e-6).all()

        assert (atomic_data[AtomicDataDict.BAND_WINDOW_KEY] == th.tensor([0, 6])).all()
        assert (atomic_data[AtomicDataDict.ENERGY_WINDOWS_KEY] == th.tensor([[-1.0, 10.0]])).all()

        assert (np.abs(atomic_data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0].numpy() 
                       - self.dataset.raw_data[0].data['eigenvalue'][0])<1e-6).all()

    def test_raw_file_names(self):
        assert self.dataset.raw_file_names == "Null"

    def test_raw_dir(self):
        assert self.dataset.raw_dir == self.rootdir

    def test_E3statistics(self):
        stats = self.dataset.E3statistics()
        assert stats is None


#TODO: Add TestDefaultDataset for E3TB. because there are some differences in the data structure.
class TestDefaultDatasetE3TB:
    def test_(self):
        pass
    def test_E3statistics(self):
        # This is the main difference between E3TB and SKTB.
        pass
