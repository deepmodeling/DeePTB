import pytest
import torch
from dptb.nnsktb.onsite_formula import onsiteFormula

def test_default_sk():
    onsiteform = onsiteFormula()
    assert onsiteform.num_paras == 0
    assert onsiteform.functype == 'none'

def test_uniform():
    mode = 'uniform'
    skform = onsiteFormula(functype=mode)
    assert skform.num_paras == 1
    
    onsite_db =  {'N':torch.tensor([1.0,2.0],dtype=torch.float64), 'B': torch.tensor([3.0,4.0],dtype=torch.float64)}
    nn_onsite_paras = {'N':torch.tensor([0.1,0.20],dtype=torch.float64), 'B': torch.tensor([0.3,0.4],dtype=torch.float64)}

    onsite_N = skform.skEs(xtype='N',onsite_db=onsite_db, nn_onsite_paras=nn_onsite_paras)
    onsite_B = skform.skEs(xtype='B',onsite_db=onsite_db, nn_onsite_paras=nn_onsite_paras)

    assert (torch.abs(onsite_N - torch.tensor([1.1,2.2],dtype=torch.float64)) < 1e-8).all()
    assert (torch.abs(onsite_B - torch.tensor([3.3,4.4],dtype=torch.float64)) < 1e-8).all()


def test_NRL_onsite():
    mode = 'NRL'
    with pytest.raises(TypeError) as exception_info:
        onsiteFormula(functype=mode,overlap=True)
    skform =  onsiteFormula(functype=mode)
    assert skform.num_paras == 4
    x_onsite_envs = torch.tensor([1.0,2.0,3.0,4.0],dtype=torch.float64)
    nn_onsite_paras= torch.tensor([[2.0, 1.0, 1.0, 1.0],[2.0, 1.0, 1.0, 1.0]])
    onsiteE = skform.skEs(x_onsite_envs=x_onsite_envs, nn_onsite_paras=nn_onsite_paras)
    assert (onsiteE - torch.tensor([3.4889900684, 3.4889900684])).abs().max() < 1e-8


def test_custom_sk():
    mode='i am not a correct name'
    with pytest.raises(ValueError) as exception_info:
        onsiteFormula(mode)

    mode ='custom'
    with pytest.raises(AssertionError) as exception_info:
        onsiteFormula(mode)