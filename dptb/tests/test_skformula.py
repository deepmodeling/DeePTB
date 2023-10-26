import pytest
import torch
from dptb.nnsktb.formula import SKFormula

def test_default_sk():
    skform = SKFormula()
    assert skform.functype == 'varTang96'
    assert skform.num_paras == 4
    assert hasattr(skform, 'varTang96')

    hij = skform.skhij(rij=1.0, paraArray=[2.0, 1.0, 1.0, 1.0])
    assert torch.abs(hij - torch.tensor([0.7357589])) < 1e-6
    hij = skform.skhij(rij=1.0, paraArray=[[2.0, 1.0, 1.0, 1.0],[2.0, -1.0, 1.0, 1.0],[2.0, -1.0, -1.0, 1.0],[2.0, -1.0, -1.0, -1.0],[-2.0, -1.0, -1.0, -1.0]])
    assert (torch.abs(hij - torch.tensor([0.7357589,0.7357589,0.7357589,0.7357589,-0.7357589])) < 1e-6).all()

def test_powerlow_sk():
    mode = 'powerlaw'
    skform = SKFormula(functype=mode)
    assert skform.num_paras == 2
    hij = skform.skhij(rij=1.0, iatomtype='Si',jatomtype='C', paraArray=[2.0, 1.0])
    hij1 = skform.skhij(rij=1.0, iatomtype='Si',jatomtype='C', paraArray=[2.0, -1.0])
    hij2 = skform.skhij(rij=1.0, iatomtype='Si',jatomtype='C', paraArray=[-2.0, -1.0])

    assert torch.abs(hij - torch.tensor([8.0872249603])) < 1e-8
    assert torch.abs(hij1 - torch.tensor([8.0872249603])) < 1e-8
    assert torch.abs(hij2 - torch.tensor([-8.0872249603])) < 1e-8

    hij  = skform.skhij(rij=1.0, iatomtype='Si',jatomtype='C', paraArray=[[2.0, 1.0],[2.0, -1.0], [-2.0, -1.0]])
    assert (torch.abs(hij - torch.tensor([8.0872249603, 8.0872249603,-8.0872249603])) < 1e-6).all()


def test_NRL_sk():
    mode = 'NRL'
    skform = SKFormula(functype=mode,overlap=False)
    assert skform.num_paras == 4
    with pytest.raises(AssertionError) as exception_info:
        assert hasattr(skform, 'overlap_num_paras')
    
    hij = skform.skhij(rij=1.0, paraArray=[2.0, 1.0, 1.0, 1.0])
    assert torch.abs(hij - torch.tensor([1.4715178013])) < 1e-8 
    hij = skform.skhij(rij=1.0, paraArray=[[2.0, 1.0, 1.0, 1.0],[2.0, 1.0, 1.0, 1.0]])
    assert (torch.abs(hij - torch.tensor([1.4715178013,1.4715178013])) < 1e-8 ).all()

    skform = SKFormula(functype=mode,overlap=True)
    assert skform.num_paras == 4
    
    hij = skform.skhij(rij=1.0, paraArray=[2.0, 1.0, 1.0, 1.0])
    assert torch.abs(hij - torch.tensor([1.4715178013])) < 1e-8 
    hij = skform.skhij(rij=1.0, paraArray=[[2.0, 1.0, 1.0, 1.0],[2.0, 1.0, 1.0, 1.0]])
    assert (torch.abs(hij - torch.tensor([1.4715178013,1.4715178013])) < 1e-8 ).all()


    assert skform.overlap_num_paras == 4

    sij = skform.sksij(rij=1.0, paraconst=[1.0], paraArray=[2.0, 1.0, 1.0, 1.0])
    assert torch.abs(sij - torch.tensor([1.8393971920])) < 1e-8 

    sij = skform.sksij(rij=1.0,paraconst=[[1.0],[0.0]], paraArray=[[2.0, 1.0, 1.0, 1.0],[2.0, 1.0, 1.0, 1.0]])
    assert (torch.abs(sij - torch.tensor([1.8393971920,1.4715178013])) < 1e-8 ).all()

def test_custom_sk():
    mode='i am not a correct name'
    with pytest.raises(ValueError) as exception_info:
        SKFormula(mode)

    mode ='custom'
    with pytest.raises(AssertionError) as exception_info:
        SKFormula(mode)