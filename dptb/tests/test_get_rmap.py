from dptb.data.AtomicData import get_r_map
from dptb.data.AtomicData import get_r_map_bondwise
import pytest
import torch

def test_get_rmap():
    r_max = {'Si': 1, 'H': 2, 'O': 4, 'C': 5}
    atomic_numbe = [1,6,8,14]
    r_map = get_r_map(r_max)

    assert r_map[1-1] == 2
    assert r_map[6-1] == 5
    assert r_map[8-1] == 4
    assert r_map[14-1] == 1

    assert isinstance(r_map, torch.Tensor)
    assert r_map.shape == (14,)

def test_get_rmap_bondwise():
    with pytest.raises(AssertionError):
        r_max = {'Si': 1, 'H': 2, 'O': 4, 'C': 5}
        get_r_map_bondwise(r_max)
    
    r_max = r_max = {'He-He': 5.5,
                      'He-H':2, 
                      'H-He':3, 
                     "Li-Li": 4,
                      "Li-H": 5,
                       'H-H': 5.5}
    r_map = get_r_map_bondwise(r_max)

    except_rmap = torch.tensor([[5.5000, 2.5000, 5.0000],
                                [2.5000, 5.5000, 0.0000],
                                [5.0000, 0.0000, 4.0000]])
    assert torch.allclose(r_map, except_rmap)
