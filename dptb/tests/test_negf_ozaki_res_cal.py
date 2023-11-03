import pytest
import torch
from dptb.negf.ozaki_res_cal import ozaki_residues

def test_ozaki():
    
    p, r = ozaki_residues(M_cut=1)
    assert abs(p-torch.tensor([3.464101615], dtype=torch.float64))<1e-8
    assert abs(r-torch.tensor([1.499999999], dtype=torch.float64))<1e-8
    p1, r1 = ozaki_residues(M_cut=2)
    for i in range(2):
        assert abs(p1[i]-torch.tensor([3.142466786, 13.043193723], dtype=torch.float64)[i])<1e-8
        assert abs(r1[i]-torch.tensor([1.002338271, 3.997661728], dtype=torch.float64)[i])<1e-8
