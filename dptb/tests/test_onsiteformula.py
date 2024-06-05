
import pytest
import torch
from dptb.nn.sktb.onsite import OnsiteFormula
from dptb.data.transforms import OrbitalMapper

basis = {"C": ['2s','2p','s*'], "Si": ['3s','3p','d*']}
idp_sk = OrbitalMapper(basis, method="sktb")
idp_sk.get_orbpair_maps()
idp_sk.get_skonsite_maps()

def test_none_onsite():
    onsite_formula = OnsiteFormula(idp=idp_sk, functype='none')
    atomic_numbers = torch.tensor([6, 14])  # Atomic number for Carbon (C) and Silicon (Si)
    onsite_energies = onsite_formula.none(atomic_numbers=atomic_numbers)
    assert onsite_energies.shape == torch.Size([2, 5])
    assert torch.all(torch.abs(onsite_energies - torch.tensor([[ 13.5727367401, 0.0, -13.6385879517,  -5.4142370224,   0.0000000000],
        [-10.8777265549, 0.0, 0.0000000000,  -4.1619720459,   0.0000000000]]))< 1e-8)

def test_uniform_onsite():
    onsite_formula = OnsiteFormula(idp=idp_sk, functype='uniform')
    atomic_numbers = torch.tensor([6, 14])  # Atomic number for Carbon (C) and Silicon (Si)
    onsite_param = torch.tensor([[[0.1],[0.0],[0.2],[0.15],[0.25]],[[0.1],[0.0],[0.2],[0.4],[0.3]]])
    onsite_energies = onsite_formula.uniform(atomic_numbers=atomic_numbers, nn_onsite_paras=onsite_param)
    assert onsite_energies.shape == torch.Size([2, 5]) 
    assert torch.all(torch.abs(onsite_energies - torch.tensor([[ 13.6727371216, 0.0, -13.4385881424,  -5.2642369270,   0.2500000000],
        [-10.7777261734,   0.0, 0.2000000030,  -3.7619719505,   0.3000000119]]))<1e-8)

