
import pytest
import torch
from dptb.nn.sktb.soc import SOCFormula
from dptb.data.transforms import OrbitalMapper

basis = {"C": ['2s','2p','s*'], "Si": ['3s','3p','d*']}
idp_sk = OrbitalMapper(basis, method="sktb")
idp_sk.get_orbpair_maps()
idp_sk.get_skonsite_maps()

def test_none_soc():
    soc_formula = SOCFormula(idp=idp_sk, functype='none')
    atomic_numbers = torch.tensor([6, 14])  # Atomic number for Carbon (C) and Silicon (Si)
    soc_Ls = soc_formula.none(atomic_numbers=atomic_numbers)
    assert soc_Ls.shape == torch.Size([2, 4]) 
    assert torch.all(torch.abs(soc_Ls - torch.tensor([[ 0.0,  0.0, 0.0, 0.0],
        [0.0,   0.0,  0.0,   0.0]]))< 1e-8)
    
    soc_Ls = soc_formula.get_socLs(atomic_numbers=atomic_numbers)
    assert soc_Ls.shape == torch.Size([2, 4])
    assert torch.all(torch.abs(soc_Ls - torch.tensor([[ 0.0,  0.0, 0.0, 0.0],
        [0.0,   0.0,  0.0,   0.0]]))< 1e-8)

def test_uniform_soc():
    soc_formula = SOCFormula(idp=idp_sk, functype='uniform')
    atomic_numbers = torch.tensor([6, 14])  # Atomic number for Carbon (C) and Silicon (Si)
    soc_param = torch.tensor([[[0.1],[0.2],[0.15],[0.25]],[[0.1],[0.2],[0.4],[0.3]]])
    soc_Ls = soc_formula.uniform(atomic_numbers=atomic_numbers, nn_soc_paras=soc_param)
    assert soc_Ls.shape == torch.Size([2, 4]) 
    assert torch.all(torch.abs(soc_Ls - torch.tensor([[0.1000, 0.2000, 0.1500, 0.2500],
                                                        [0.1000, 0.2000, 0.4000, 0.3000]]))<1e-8)
    
    soc_param = torch.tensor([[[-0.1],[0.2],[0.15],[0.25]],[[0.1],[0.2],[0.4],[-0.3]]])
    soc_Ls = soc_formula.get_socLs(atomic_numbers=atomic_numbers, nn_soc_paras=soc_param)
    assert soc_Ls.shape == torch.Size([2, 4]) 
    assert torch.all(torch.abs(soc_Ls - torch.tensor([[0.1000, 0.2000, 0.1500, 0.2500],
                                                        [0.1000, 0.2000, 0.4000, 0.3000]]))<1e-8)