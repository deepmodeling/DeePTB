import os
import pytest
import torch
from pathlib import Path
from dptb.data import AtomicDataset, DataLoader, AtomicDataDict,AtomicData
from dptb.data.build import build_dataset
from dptb.data.dataloader import Collater
from dptb.utils.torch_geometric.batch import Batch
from dptb.utils.torch_geometric.data import Data
from collections.abc import Mapping
from dptb.tests.tstools import compare_tensors_as_sets_float

rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")


class TestDataLoaderBatch:
    data_options = {
        "r_max": 5.0,
        "er_max": 5.0,
        "oer_max": 2.5,
        "train": {
            "root": f"{rootdir}/test_sktb/dataset",
            "prefix": "kpath_spk",
            "get_eigenvalues": True
        }
    }
    common_options = {
    "basis": {
        "Si": ["3s","3p"]
    },
    "device": "cpu",
    "dtype": "float32",
    "overlap": False,
    "seed": 3982377700
    }
    train_datasets = build_dataset(**data_options, **data_options["train"], **common_options)

    def test_init(self):
        train_loader = DataLoader(dataset=self.train_datasets, batch_size=1, shuffle=True)
        assert isinstance(train_loader, DataLoader)
        assert isinstance(train_loader.dataset, AtomicDataset)
        assert isinstance(train_loader.collate_fn, Collater)
        assert train_loader.batch_size == 1
        assert train_loader.drop_last == False
        assert train_loader.num_workers == 0
        assert train_loader.pin_memory == False
        assert train_loader.timeout == 0
        assert train_loader.worker_init_fn == None
        assert train_loader.multiprocessing_context == None
        assert train_loader.generator == None
        assert train_loader.collate_fn.exclude_keys == []
    
    def test_batch(self):
        train_loader = DataLoader(dataset=self.train_datasets, batch_size=1, shuffle=True)
        for batch in train_loader:
            assert isinstance(batch, Batch)
            assert isinstance(batch, Data)
            assert batch.num_graphs == 1
            assert batch.num_nodes == 2
            assert batch.num_edges == 56
            assert batch.x  is None
            assert batch.edge_index.shape == torch.Size([2, 56])
            assert batch.edge_attr is None
            break

        batch = AtomicData.to_AtomicDataDict(batch)
        assert isinstance(batch, dict)
        assert isinstance(batch, Mapping)

        assert torch.all(batch[AtomicDataDict.ATOM_TYPE_KEY] ==  torch.tensor([[0],[0]]))
        with pytest.raises(KeyError):
            batch[AtomicDataDict.EDGE_LENGTH_KEY]
        with pytest.raises(KeyError):
            batch[AtomicDataDict.EDGE_VECTORS_KEY]

        batch = AtomicDataDict.with_edge_vectors(batch, with_lengths=True)
        assert batch[AtomicDataDict.EDGE_LENGTH_KEY].shape == torch.Size([56])
        assert batch[AtomicDataDict.EDGE_TYPE_KEY].shape == torch.Size([56, 1])

        expected_length = torch.tensor([3.8395895958, 4.5023179054, 3.8395895958, 4.5023179054, 3.8395895958,
                                        4.5023179054, 3.8395895958, 2.3512587547, 4.5023179054, 2.3512589931,
                                        4.5023179054, 4.5023179054, 3.8395895958, 4.5023179054, 3.8395895958,
                                        2.3512587547, 4.5023179054, 4.5023179054, 4.5023179054, 2.3512587547,
                                        4.5023179054, 4.5023179054, 3.8395895958, 3.8395895958, 3.8395895958,
                                        3.8395895958, 3.8395895958, 3.8395895958, 3.8395895958, 4.5023179054,
                                        3.8395895958, 4.5023179054, 3.8395895958, 4.5023179054, 3.8395895958,
                                        2.3512587547, 4.5023179054, 2.3512589931, 4.5023179054, 4.5023179054,
                                        3.8395895958, 4.5023179054, 3.8395895958, 2.3512587547, 4.5023179054,
                                        4.5023179054, 4.5023179054, 2.3512587547, 4.5023179054, 4.5023179054,
                                        3.8395895958, 3.8395895958, 3.8395895958, 3.8395895958, 3.8395895958,
                                        3.8395895958])
        assert compare_tensors_as_sets_float(batch[AtomicDataDict.EDGE_LENGTH_KEY], expected_length, precision=7)
        # assert torch.all(torch.abs(batch[AtomicDataDict.EDGE_LENGTH_KEY] - expected_length) < 1e-8)
        
        assert batch[AtomicDataDict.EDGE_VECTORS_KEY].shape == torch.Size([56, 3])
        expected_edgevectors = torch.tensor([[ 1.9197947979,  3.3251821995,  0.0000000000],
                                             [ 0.0000000000,  4.4335761070,  0.7837529182],
                                             [-1.9197947979,  3.3251821995,  0.0000000000],
                                             [ 1.9197947979, -1.1083942652,  3.9187645912],
                                             [ 0.0000000000, -2.2167882919,  3.1350116730],
                                             [-1.9197945595, -1.1083942652,  3.9187645912],
                                             [ 3.8395895958,  0.0000000000,  0.0000000000],
                                             [-1.9197947979,  1.1083940268,  0.7837529182],
                                             [ 3.8395895958, -2.2167882919,  0.7837529182],
                                             [ 0.0000000000, -2.2167882919,  0.7837529182],
                                             [-3.8395893574, -2.2167882919,  0.7837529182],
                                             [ 1.9197947979,  3.3251824379, -2.3512587547],
                                             [ 1.9197947979,  1.1083940268,  3.1350116730],
                                             [ 0.0000000000,  2.2167880535,  3.9187645912],
                                             [-1.9197947979,  1.1083940268,  3.1350116730],
                                             [ 1.9197947979,  1.1083940268,  0.7837529182],
                                             [ 1.9197947979, -3.3251819611, -2.3512587547],
                                             [-1.9197947979, -3.3251819611, -2.3512587547],
                                             [-1.9197947979,  3.3251824379, -2.3512587547],
                                             [ 0.0000000000,  0.0000000000, -2.3512587547],
                                             [-3.8395893574,  0.0000000000, -2.3512587547],
                                             [ 3.8395895958,  0.0000000000, -2.3512587547],
                                             [-1.9197947979, -3.3251821995,  0.0000000000],
                                             [ 1.9197947979, -3.3251821995,  0.0000000000],
                                             [ 3.8395895958,  0.0000000000,  0.0000000000],
                                             [ 1.9197947979, -1.1083940268, -3.1350116730],
                                             [ 0.0000000000,  2.2167882919, -3.1350116730],
                                             [-1.9197947979, -1.1083940268, -3.1350116730],
                                             [-1.9197947979, -3.3251821995,  0.0000000000],
                                             [ 0.0000000000, -4.4335761070, -0.7837529182],
                                             [ 1.9197947979, -3.3251821995,  0.0000000000],
                                             [-1.9197947979,  1.1083942652, -3.9187645912],
                                             [ 0.0000000000,  2.2167882919, -3.1350116730],
                                             [ 1.9197945595,  1.1083942652, -3.9187645912],
                                             [-3.8395895958,  0.0000000000,  0.0000000000],
                                             [ 1.9197947979, -1.1083940268, -0.7837529182],
                                             [-3.8395895958,  2.2167882919, -0.7837529182],
                                             [ 0.0000000000,  2.2167882919, -0.7837529182],
                                             [ 3.8395893574,  2.2167882919, -0.7837529182],
                                             [-1.9197947979, -3.3251824379,  2.3512587547],
                                             [-1.9197947979, -1.1083940268, -3.1350116730],
                                             [ 0.0000000000, -2.2167880535, -3.9187645912],
                                             [ 1.9197947979, -1.1083940268, -3.1350116730],
                                             [-1.9197947979, -1.1083940268, -0.7837529182],
                                             [-1.9197947979,  3.3251819611,  2.3512587547],
                                             [ 1.9197947979,  3.3251819611,  2.3512587547],
                                             [ 1.9197947979, -3.3251824379,  2.3512587547],
                                             [ 0.0000000000,  0.0000000000,  2.3512587547],
                                             [ 3.8395893574,  0.0000000000,  2.3512587547],
                                             [-3.8395895958,  0.0000000000,  2.3512587547],
                                             [ 1.9197947979,  3.3251821995,  0.0000000000],
                                             [-1.9197947979,  3.3251821995,  0.0000000000],
                                             [-3.8395895958,  0.0000000000,  0.0000000000],
                                             [-1.9197947979,  1.1083940268,  3.1350116730],
                                             [ 0.0000000000, -2.2167882919,  3.1350116730],
                                             [ 1.9197947979,  1.1083940268,  3.1350116730]])
        assert compare_tensors_as_sets_float(batch[AtomicDataDict.EDGE_VECTORS_KEY], expected_edgevectors, precision=7)
        # assert torch.all(torch.abs(batch[AtomicDataDict.EDGE_VECTORS_KEY] - expected_edgevectors) < 1e-8)

        
        batch = AtomicDataDict.with_env_vectors(batch, with_lengths=True)
        assert batch[AtomicDataDict.ENV_LENGTH_KEY].shape == torch.Size([56])
        assert batch[AtomicDataDict.ENV_INDEX_KEY].shape == torch.Size([2,56])
        expected_env_index = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                                            1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1, 1, 1],
                                           [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 1, 1, 1, 1, 1, 1]])
        

        expected_env_length = torch.tensor([3.8395895958, 4.5023179054, 3.8395895958, 4.5023179054, 3.8395895958,
        4.5023179054, 3.8395895958, 2.3512587547, 4.5023179054, 2.3512589931,
        4.5023179054, 4.5023179054, 3.8395895958, 4.5023179054, 3.8395895958,
        2.3512587547, 4.5023179054, 4.5023179054, 4.5023179054, 2.3512587547,
        4.5023179054, 4.5023179054, 3.8395895958, 3.8395895958, 3.8395895958,
        3.8395895958, 3.8395895958, 3.8395895958, 3.8395895958, 4.5023179054,
        3.8395895958, 4.5023179054, 3.8395895958, 4.5023179054, 3.8395895958,
        2.3512587547, 4.5023179054, 2.3512589931, 4.5023179054, 4.5023179054,
        3.8395895958, 4.5023179054, 3.8395895958, 2.3512587547, 4.5023179054,
        4.5023179054, 4.5023179054, 2.3512587547, 4.5023179054, 4.5023179054,
        3.8395895958, 3.8395895958, 3.8395895958, 3.8395895958, 3.8395895958,
        3.8395895958])



        expected_env_vectors =  torch.tensor([[ 1.9197947979,  3.3251821995,  0.0000000000],
        [ 0.0000000000,  4.4335761070,  0.7837529182],
        [-1.9197947979,  3.3251821995,  0.0000000000],
        [ 1.9197947979, -1.1083942652,  3.9187645912],
        [ 0.0000000000, -2.2167882919,  3.1350116730],
        [-1.9197945595, -1.1083942652,  3.9187645912],
        [ 3.8395895958,  0.0000000000,  0.0000000000],
        [-1.9197947979,  1.1083940268,  0.7837529182],
        [ 3.8395895958, -2.2167882919,  0.7837529182],
        [ 0.0000000000, -2.2167882919,  0.7837529182],
        [-3.8395893574, -2.2167882919,  0.7837529182],
        [ 1.9197947979,  3.3251824379, -2.3512587547],
        [ 1.9197947979,  1.1083940268,  3.1350116730],
        [ 0.0000000000,  2.2167880535,  3.9187645912],
        [-1.9197947979,  1.1083940268,  3.1350116730],
        [ 1.9197947979,  1.1083940268,  0.7837529182],
        [ 1.9197947979, -3.3251819611, -2.3512587547],
        [-1.9197947979, -3.3251819611, -2.3512587547],
        [-1.9197947979,  3.3251824379, -2.3512587547],
        [ 0.0000000000,  0.0000000000, -2.3512587547],
        [-3.8395893574,  0.0000000000, -2.3512587547],
        [ 3.8395895958,  0.0000000000, -2.3512587547],
        [-1.9197947979, -3.3251821995,  0.0000000000],
        [ 1.9197947979, -3.3251821995,  0.0000000000],
        [ 3.8395895958,  0.0000000000,  0.0000000000],
        [ 1.9197947979, -1.1083940268, -3.1350116730],
        [ 0.0000000000,  2.2167882919, -3.1350116730],
        [-1.9197947979, -1.1083940268, -3.1350116730],
        [-1.9197947979, -3.3251821995,  0.0000000000],
        [ 0.0000000000, -4.4335761070, -0.7837529182],
        [ 1.9197947979, -3.3251821995,  0.0000000000],
        [-1.9197947979,  1.1083942652, -3.9187645912],
        [ 0.0000000000,  2.2167882919, -3.1350116730],
        [ 1.9197945595,  1.1083942652, -3.9187645912],
        [-3.8395895958,  0.0000000000,  0.0000000000],
        [ 1.9197947979, -1.1083940268, -0.7837529182],
        [-3.8395895958,  2.2167882919, -0.7837529182],
        [ 0.0000000000,  2.2167882919, -0.7837529182],
        [ 3.8395893574,  2.2167882919, -0.7837529182],
        [-1.9197947979, -3.3251824379,  2.3512587547],
        [-1.9197947979, -1.1083940268, -3.1350116730],
        [ 0.0000000000, -2.2167880535, -3.9187645912],
        [ 1.9197947979, -1.1083940268, -3.1350116730],
        [-1.9197947979, -1.1083940268, -0.7837529182],
        [-1.9197947979,  3.3251819611,  2.3512587547],
        [ 1.9197947979,  3.3251819611,  2.3512587547],
        [ 1.9197947979, -3.3251824379,  2.3512587547],
        [ 0.0000000000,  0.0000000000,  2.3512587547],
        [ 3.8395893574,  0.0000000000,  2.3512587547],
        [-3.8395895958,  0.0000000000,  2.3512587547],
        [ 1.9197947979,  3.3251821995,  0.0000000000],
        [-1.9197947979,  3.3251821995,  0.0000000000],
        [-3.8395895958,  0.0000000000,  0.0000000000],
        [-1.9197947979,  1.1083940268,  3.1350116730],
        [ 0.0000000000, -2.2167882919,  3.1350116730],
        [ 1.9197947979,  1.1083940268,  3.1350116730]])

        expect_envs = torch.cat([expected_env_index.T, expected_env_length.unsqueeze(1), expected_env_vectors], dim=1)
        target_envs = torch.cat([batch[AtomicDataDict.ENV_INDEX_KEY].T, batch[AtomicDataDict.ENV_LENGTH_KEY].unsqueeze(1), batch[AtomicDataDict.ENV_VECTORS_KEY]], dim=1)
        assert compare_tensors_as_sets_float(target_envs, expect_envs, precision=7)
        
        #assert torch.all(torch.abs(batch[AtomicDataDict.ENV_VECTORS_KEY] - expected_env_vectors) < 1e-8)
        #assert torch.all(batch[AtomicDataDict.ENV_INDEX_KEY] == expected_env_index)
        #assert torch.all(torch.abs(batch[AtomicDataDict.ENV_LENGTH_KEY] - expected_env_length) < 1e-8)

        batch = AtomicDataDict.with_onsitenv_vectors(batch, with_lengths=True)
        assert batch[AtomicDataDict.ONSITENV_INDEX_KEY].shape == torch.Size([2, 8])
        assert batch[AtomicDataDict.ONSITENV_LENGTH_KEY].shape == torch.Size([8])
        
        expected_onsiteenv_index = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0]])
        expected_onsiteenv_length = torch.tensor([2.3512587547, 2.3512589931, 2.3512587547, 2.3512587547, 2.3512587547,
        2.3512589931, 2.3512587547, 2.3512587547])
        expected_onsiteenv_vectors = torch.tensor([[-1.9197947979,  1.1083940268,  0.7837529182],
        [ 0.0000000000, -2.2167882919,  0.7837529182],
        [ 1.9197947979,  1.1083940268,  0.7837529182],
        [ 0.0000000000,  0.0000000000, -2.3512587547],
        [ 1.9197947979, -1.1083940268, -0.7837529182],
        [ 0.0000000000,  2.2167882919, -0.7837529182],
        [-1.9197947979, -1.1083940268, -0.7837529182],
        [ 0.0000000000,  0.0000000000,  2.3512587547]])

        expected_onsiteenvs = torch.cat([expected_onsiteenv_index.T, expected_onsiteenv_length.unsqueeze(1), expected_onsiteenv_vectors], dim=1)
        target_onsiteenvs = torch.cat([batch[AtomicDataDict.ONSITENV_INDEX_KEY].T, batch[AtomicDataDict.ONSITENV_LENGTH_KEY].unsqueeze(1), batch[AtomicDataDict.ONSITENV_VECTORS_KEY]], dim=1)
        assert compare_tensors_as_sets_float(target_onsiteenvs, expected_onsiteenvs, precision=7)

        #assert torch.all(batch[AtomicDataDict.ONSITENV_INDEX_KEY] == expected_onsiteenv_index)
        #assert torch.all(torch.abs(batch[AtomicDataDict.ONSITENV_LENGTH_KEY] - expected_onsiteenv_length) < 1e-8)
        #assert torch.all(torch.abs(batch[AtomicDataDict.ONSITENV_VECTORS_KEY] - expected_onsiteenv_vectors) < 1e-8)