import pytest
import os 
from pathlib import Path
import numpy as np
from dptb.nn.build import build_model
from ase.io import read
from dptb.data import AtomicData, AtomicDataDict
import torch

rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")

class TestEmbSe2:
    ckpt = f'{rootdir}/mos2/mix.ep500.pth'
    model = build_model(checkpoint=ckpt)
    stru_data = f'{rootdir}/mos2/struct.vasp'
    AtomicData_options={"r_max": 5.0,
            "er_max": 3.5,
            "oer_max":1.6,
            "pbc": True
            }
    structase = read(stru_data)

    data = AtomicData.from_ase(structase, **AtomicData_options)
    data = AtomicData.to_AtomicDataDict(data)
    data = model.idp(data)

    data = model.nnenv.embedding(data)

    def test_embedding(self):
        data = self.data
        env_vectors = data['env_vectors']
        env_index   = data['env_index']
        atom_attr   = data['node_attrs']
        edge_index  = data['edge_index']
        edge_length = data['edge_lengths']
        n_env = env_index.shape[1]
        env_attr = atom_attr[env_index].transpose(1,0).reshape(n_env,-1)
        
        iind = env_index[0] 
        jind = env_index[1] 


        for ii in [0,1,2]:
            for jj in [0,1,2]:
                btyatt = torch.cat([atom_attr[ii],atom_attr[jj]],dim=0)
                assert torch.all(env_attr[(iind == ii) * (jind == jj)] == btyatt)
    
        se2 = self.model.nnenv.embedding.descriptor

        assert se2.decomposed_layers == 1
        assert se2.explain in [None, False]
        kwargs={'env_vectors':env_vectors, 'env_attr':env_attr}
        size = se2._check_input(env_index, None)
        assert size == [None, None]
        coll_dict = se2._collect(se2._user_args, env_index, size,kwargs)
        
        for ikey in ['env_attr', 'env_vectors', 'adj_t', 
                     'edge_index', 'edge_index_i', 'edge_index_j', 
                     'ptr', 'index', 'size', 'size_i', 'size_j', 'dim_size']:

            assert ikey in coll_dict.keys()
        
        assert torch.all(coll_dict['env_attr'] == torch.tensor([
                                                [0., 1., 0., 1.],[0., 1., 0., 1.],[0., 1., 1., 0.],[0., 1., 0., 1.],[0., 1., 1., 0.],[0., 1., 1., 0.],[0., 1., 1., 0.],[0., 1., 1., 0.],[0., 1., 1., 0.],[1., 0., 1., 0.],
                                                [1., 0., 1., 0.],[1., 0., 1., 0.],[1., 0., 1., 0.],[1., 0., 1., 0.],[1., 0., 1., 0.],[1., 0., 1., 0.],[0., 1., 0., 1.],[0., 1., 0., 1.],[1., 0., 0., 1.],[0., 1., 0., 1.],
                                                [1., 0., 0., 1.],[1., 0., 0., 1.],[1., 0., 0., 1.],[1., 0., 0., 1.],[1., 0., 0., 1.],[1., 0., 1., 0.],[1., 0., 1., 0.],[1., 0., 1., 0.],[1., 0., 1., 0.],[1., 0., 1., 0.],
                                                [1., 0., 1., 0.],[1., 0., 1., 0.]]))

        assert torch.all(coll_dict['env_vectors'] == env_vectors)
        assert torch.all(coll_dict['edge_index'] == env_index)
        assert torch.all(coll_dict['edge_index_i'] == env_index[0])
        assert torch.all(coll_dict['edge_index_j'] == env_index[1])
        assert torch.all(coll_dict['index'] == env_index[0])

        msg_kwargs = se2.inspector.distribute('message', coll_dict)
        assert list(msg_kwargs.keys()) == ['env_vectors', 'env_attr']

        assert torch.all(msg_kwargs['env_vectors'] == env_vectors)
        assert torch.all(msg_kwargs['env_attr'] == coll_dict['env_attr'])

        assert len(se2._message_forward_pre_hooks.values()) == 0
        assert len(se2._message_forward_hooks.values()) == 0
        assert len(se2._aggregate_forward_pre_hooks.values()) == 0
        assert len(se2._aggregate_forward_hooks.values()) == 0
        assert len(se2._propagate_forward_hooks.values()) == 0


        inp1 = torch.tensor([[1./2,0,1,0,1]])
        inp2 = torch.tensor([[1./2,0,1,0,1]])
        assert torch.allclose(se2.embedding_net(inp1), se2.embedding_net(inp2))

        inp2 = torch.tensor([[1./2,1,0,0,1]])
        assert not torch.allclose(se2.embedding_net(inp1), se2.embedding_net(inp2))

        inp1 = torch.tensor([[1./2,0,1,1,0]])
        assert not torch.allclose(se2.embedding_net(inp1), se2.embedding_net(inp2))

        inp1 = torch.tensor([[1./2,0,1,0,1]])
        inp2 = torch.tensor([[1./2,1,0,1,0]])
        assert not torch.allclose(se2.embedding_net(inp1), se2.embedding_net(inp2))

        expected_values = torch.tensor([[ 0.52560139,  0.34891951,  0.13165946,  0.47105983,  0.07533115,
          0.18981691, -0.37594870,  0.18878269,  0.45241335, -0.03054990]])
        
        assert torch.allclose(se2.embedding_net(inp2), expected_values)


        rij = env_vectors.norm(dim=-1, keepdim=True)
        snorm = se2.smooth(rij, se2.rs, se2.rc)
        
        
        emb_Gmat = se2.embedding_net(torch.cat([snorm, env_attr], dim=-1))
        assert emb_Gmat.shape == torch.Size([32, 10])

        env_vectors_new = torch.cat([snorm, snorm * env_vectors / rij], dim=-1)
        emv_cat = torch.cat([emb_Gmat, env_vectors_new], dim=-1) # [N_env, D_emb + 4]

        out = se2.message(**msg_kwargs)
        assert torch.allclose(out, emv_cat)

        aggr_kwargs = se2.inspector.distribute('aggregate', coll_dict)
        assert torch.all(aggr_kwargs['index'] == coll_dict['index'])

        out2 = se2.aggregate(out, **aggr_kwargs)
        vect = out[:,-4:]
        emvg = out[:,:-4]
        indx = aggr_kwargs['index']
        for i in range(3):
            re1 = emvg[indx==i].T @ vect[indx==i]/ vect[indx==i].shape[0]
            assert torch.all(re1==out2[i])

        update_kwargs = se2.inspector.distribute('update', coll_dict)
        assert len(update_kwargs) == 0


        out = se2.update(out2, **update_kwargs)
        for ii in range(3):
            re1 = (out2[ii] @ out2[ii].T).reshape(-1)
            re1 -= re1.mean()
            if ii==0:
                assert torch.abs(re1.norm() - 0.06084440) < 1e-6
            assert re1.norm() > 1e-6    
            re1 /= re1.norm()
            assert torch.all(out[ii] == re1)

        assert len(se2._edge_update_forward_pre_hooks.values()) == 0
        size = se2._check_input(edge_index, size=None)
        assert size == [None, None]
        
        se2._edge_user_args == {'edge_length', 'node_descriptor'}
        kwargs={'edge_length':edge_length, 'node_descriptor':out}
        coll_dict = se2._collect(se2._edge_user_args, edge_index, size, kwargs)

        for ikey in ['edge_length',
                      'node_descriptor',
                      'adj_t',
                      'edge_index',
                      'edge_index_i',
                      'edge_index_j',
                      'ptr',
                      'index',
                      'size',
                      'size_i',
                      'size_j',
                      'dim_size']:
            assert ikey in coll_dict.keys()
        
        assert torch.all(coll_dict['node_descriptor'] == out)
        assert torch.all(coll_dict['edge_length'] == edge_length)
        assert torch.all(coll_dict['edge_index'] == edge_index)
        assert torch.all(coll_dict['edge_index_i'] == edge_index[0])
        assert torch.all(coll_dict['edge_index_j'] == edge_index[1])
        assert torch.all(coll_dict['index'] == edge_index[0])

        edge_kwargs = se2.inspector.distribute('edge_update', coll_dict)
        
        assert list(edge_kwargs.keys()) == ['edge_index', 'node_descriptor', 'edge_length']

        assert torch.all(edge_kwargs['edge_index'] == edge_index)
        assert torch.all(edge_kwargs['node_descriptor'] == out)
        assert torch.all(edge_kwargs['edge_length'] == edge_length)
        
        edge_out = torch.cat([out[edge_index[0]] + out[edge_index[1]], 1/edge_length.reshape(-1,1)], dim=-1) # [N_edge, D*D]
        
        out = se2.edge_update(**edge_kwargs)

        assert torch.all(out==edge_out)
        assert out.shape == torch.Size([56, 101])
    
    def test_smooth(self):
        se2 = self.model.nnenv.embedding.descriptor
        with pytest.raises(AssertionError):
            se2.smooth(torch.tensor(1.0), 4.5, 3.5)
        
        rr = torch.linspace(0.1,5,30,dtype=torch.float32)
        
        expected_values = torch.tensor([1.00000000e+01, 3.71794868e+00, 2.28346467e+00, 1.64772725e+00,
                                        1.28888881e+00, 1.05839419e+00, 8.97832811e-01, 7.79569924e-01,
                                        6.88836098e-01, 6.17021263e-01, 5.58766842e-01, 5.10563374e-01,
                                        4.70016211e-01, 4.35435444e-01, 4.05594409e-01, 3.72111082e-01,
                                        2.96894461e-01, 1.85579583e-01, 7.91704208e-02, 1.51895694e-02,
                                        2.45147003e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                        0.00000000e+00, 0.00000000e+00])

        assert torch.allclose(se2.smooth(rr, 2.5, 3.5), expected_values)

