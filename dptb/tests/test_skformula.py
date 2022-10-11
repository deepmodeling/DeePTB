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
    hij = skform.skhij(rij=1.0, paraArray=[[2.0, 1.0, 1.0, 1.0],[2.0, 1.0, 1.0, 1.0]])
    assert (torch.abs(hij - torch.tensor([0.7357589,0.7357589])) < 1e-6).all()

def test_custom_sk():
    mode='i am not a correct name'
    with pytest.raises(ValueError) as exception_info:
        SKFormula(mode)

    mode ='custom'
    with pytest.raises(AssertionError) as exception_info:
        SKFormula(mode)