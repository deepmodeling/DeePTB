import pytest
from dptb.data.build import build_dataset
from dptb.data.dataset import DefaultDataset
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataset, DataLoader, AtomicDataDict,AtomicData
import os
from pathlib import Path
import torch
from dptb.nn.nnsk import NNSK
from dptb.utils.torch_geometric import Batch


rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")


class TestMultiBatch:
    set_options = {
    "r_max": 5.0,
    "er_max": 5.0,
    "oer_max": 2.5,
    "root": f"{rootdir}/test_sktb/dataset",
    "prefix": "kpathmd25",
    "get_eigenvalues": True,
    "get_Hamiltonian": False,
    }
    common_options={"basis": {"Si": ["3s", "3p"]}}
    dataset = build_dataset(**set_options, **common_options)

    dload= DataLoader(dataset, batch_size=2, shuffle=False)
    dload2= DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(dload))

    batch_info = {
    "__slices__": batch.__slices__,
    "__cumsum__": batch.__cumsum__,
    "__cat_dims__": batch.__cat_dims__,
    "__num_nodes_list__": batch.__num_nodes_list__,
    "__data_class__": batch.__data_class__,
    }
    
    batch2s = []
    ic=0
    for ibatch in dload2:
        ic +=1
        batch2s.append(ibatch)
        if ic>2:
            break
    batch21 = batch2s[0]
    batch22 = batch2s[1]

    def test_load(self):
        batch_dict = AtomicData.to_AtomicDataDict(self.batch)
        batch21_dict = AtomicData.to_AtomicDataDict(self.batch21)
        batch22_dict = AtomicData.to_AtomicDataDict(self.batch22)

        assert batch_dict['edge_index'].shape == torch.Size([2, 448])
        assert batch21_dict['edge_index'].shape == torch.Size([2, 224])
        assert batch22_dict['edge_index'].shape == torch.Size([2, 224])

        assert torch.all(batch_dict['edge_index'][:,:224] == batch21_dict['edge_index'])
        assert torch.all(batch_dict['edge_index'][:,224:]-8 == batch22_dict['edge_index'])

        assert batch_dict['pos'].shape == torch.Size([16, 3])
        assert batch21_dict['pos'].shape == torch.Size([8, 3])
        assert batch22_dict['pos'].shape == torch.Size([8, 3])

        assert torch.allclose(batch_dict['pos'], torch.cat([batch21_dict['pos'], batch22_dict['pos']]))

        assert torch.allclose(batch_dict['batch'], torch.cat([batch21_dict['batch'], batch22_dict['batch']+1]))

        assert torch.all(batch_dict['ptr'] == torch.tensor([0, 8, 16]))
        assert torch.all(batch21_dict['ptr'] == torch.tensor([0, 8]))
        assert torch.all(batch22_dict['ptr'] == torch.tensor([0, 8]))

        assert batch_dict['env_index'].shape == torch.Size([2, 448])
        assert batch21_dict['env_index'].shape == torch.Size([2, 224])
        assert batch22_dict['env_index'].shape == torch.Size([2, 224])

        assert torch.all(batch_dict['env_index'][:,:224] == batch21_dict['env_index'])
        assert torch.all(batch_dict['env_index'][:,224:]-8 == batch22_dict['env_index'])


        assert torch.all(batch_dict['env_cell_shift'] == torch.cat([batch21_dict['env_cell_shift'], batch22_dict['env_cell_shift']]))
        assert torch.all(batch_dict['edge_cell_shift'] == torch.cat([batch21_dict['edge_cell_shift'], batch22_dict['edge_cell_shift']]))
        assert torch.all(batch_dict['onsitenv_cell_shift'] == torch.cat([batch21_dict['onsitenv_cell_shift'], batch22_dict['onsitenv_cell_shift']]))

        assert batch_dict['onsitenv_index'].shape == torch.Size([2, 64])
        assert batch21_dict['onsitenv_index'].shape == torch.Size([2, 32])
        assert batch22_dict['onsitenv_index'].shape == torch.Size([2, 32])

        assert torch.all(batch_dict['onsitenv_index'][:,:32] == batch21_dict['onsitenv_index'])
        assert torch.all(batch_dict['onsitenv_index'][:,32:]-8 == batch22_dict['onsitenv_index'])

    def test_model_update(self):
        batch_dict = AtomicData.to_AtomicDataDict(self.batch)
        batch21_dict = AtomicData.to_AtomicDataDict(self.batch21)
        batch22_dict = AtomicData.to_AtomicDataDict(self.batch22)

        model = NNSK.from_reference(checkpoint=f'{rootdir}/test_sktb/output/test_push_w/checkpoint/nnsk.best.pth')
        model.transform = False
        batch_dict = model(batch_dict)
        batch21_dict = model(batch21_dict)
        batch22_dict = model(batch22_dict)

        assert torch.allclose(batch_dict['node_features'], torch.cat([batch21_dict['node_features'], batch22_dict['node_features']]))
        assert torch.allclose(batch_dict['edge_features'], torch.cat([batch21_dict['edge_features'], batch22_dict['edge_features']]))


        model.transform = True
        assert torch.allclose(batch_dict['node_features'], torch.cat([batch21_dict['node_features'], batch22_dict['node_features']]))
        assert torch.allclose(batch_dict['edge_features'], torch.cat([batch21_dict['edge_features'], batch22_dict['edge_features']]))


    def test_tolist(self):
        
        batch_dict = AtomicData.to_AtomicDataDict(self.batch)
        batch21_dict = AtomicData.to_AtomicDataDict(self.batch21)
        batch22_dict = AtomicData.to_AtomicDataDict(self.batch22)

        model = NNSK.from_reference(checkpoint=f'{rootdir}/test_sktb/output/test_push_w/checkpoint/nnsk.best.pth')
        batch_dict = model(batch_dict)
        batch21_dict = model(batch21_dict)
        batch22_dict = model(batch22_dict)

        batch_dict.update(self.batch_info)
        batch_dict = Batch.from_dict(batch_dict)
        batch_list = batch_dict.to_data_list()
        batch11, batch12 = batch_list[0], batch_list[1]

        batch11_dict = AtomicData.to_AtomicDataDict(batch11)
        batch12_dict = AtomicData.to_AtomicDataDict(batch12)

    
        assert batch11_dict['edge_index'].shape == torch.Size([2, 224])
        assert batch12_dict['edge_index'].shape == torch.Size([2, 224])
        assert batch21_dict['edge_index'].shape == torch.Size([2, 224])
        assert batch22_dict['edge_index'].shape == torch.Size([2, 224])

        assert torch.all(batch11_dict['edge_index'] == batch21_dict['edge_index'])
        assert torch.all(batch12_dict['edge_index'] == batch22_dict['edge_index'])

        assert batch11_dict['pos'].shape == torch.Size([8, 3])
        assert batch12_dict['pos'].shape == torch.Size([8, 3])
        assert batch21_dict['pos'].shape == torch.Size([8, 3])
        assert batch22_dict['pos'].shape == torch.Size([8, 3])

        assert torch.allclose(batch11_dict['pos'], batch21_dict['pos'])
        assert torch.allclose(batch12_dict['pos'], batch22_dict['pos'])

        #assert torch.allclose(batch11_dict['batch'], batch21_dict['batch'])
        #assert torch.allclose(batch12_dict['batch'], batch22_dict['batch'])

        #assert torch.all(batch11_dict['ptr'] == torch.tensor([0, 8]))
        #assert torch.all(batch12_dict['ptr'] == torch.tensor([0, 8]))
        #assert torch.all(batch21_dict['ptr'] == torch.tensor([0, 8]))
        #assert torch.all(batch22_dict['ptr'] == torch.tensor([0, 8]))

        assert batch11_dict['env_index'].shape == torch.Size([2, 224])
        assert batch12_dict['env_index'].shape == torch.Size([2, 224])
        assert batch21_dict['env_index'].shape == torch.Size([2, 224])
        assert batch22_dict['env_index'].shape == torch.Size([2, 224])

        assert torch.all(batch11_dict['env_index'] == batch21_dict['env_index'])
        assert torch.all(batch12_dict['env_index'] == batch22_dict['env_index'])


        assert torch.all(batch11_dict['env_cell_shift'] == batch21_dict['env_cell_shift'])
        assert torch.all(batch12_dict['env_cell_shift'] == batch22_dict['env_cell_shift'])
        assert torch.all(batch11_dict['edge_cell_shift'] == batch21_dict['edge_cell_shift'])
        assert torch.all(batch12_dict['edge_cell_shift'] == batch22_dict['edge_cell_shift'])
        assert torch.all(batch11_dict['onsitenv_cell_shift'] == batch21_dict['onsitenv_cell_shift'])
        assert torch.all(batch12_dict['onsitenv_cell_shift'] == batch22_dict['onsitenv_cell_shift'])

        assert batch11_dict['onsitenv_index'].shape == torch.Size([2, 32])
        assert batch12_dict['onsitenv_index'].shape == torch.Size([2, 32])
        assert batch21_dict['onsitenv_index'].shape == torch.Size([2, 32])
        assert batch22_dict['onsitenv_index'].shape == torch.Size([2, 32])

        assert torch.all(batch11_dict['onsitenv_index'] == batch21_dict['onsitenv_index'])
        assert torch.all(batch12_dict['onsitenv_index'] == batch22_dict['onsitenv_index'])

        assert torch.allclose(batch11_dict['node_features'], batch21_dict['node_features'])
        assert torch.allclose(batch12_dict['node_features'], batch22_dict['node_features'])
        
        assert torch.allclose(batch11_dict['edge_features'], batch21_dict['edge_features'])
        assert torch.allclose(batch12_dict['edge_features'], batch22_dict['edge_features'])
