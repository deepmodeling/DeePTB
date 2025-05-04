# 基于pytest 生成对 auto_band_config.py 的测试
import pytest
import os
from pathlib import Path
from dptb.utils.auto_band_config import auto_band_config

rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")

def test_auto_band_config():
    try:
        import seekpath
    except:
        pytest.skip("Seekpath is not installed in the current image, will skip this test.")
    structure = f'{rootdir}/mos2/struct.vasp'
    
    bandjdata, common_options = auto_band_config(structure=structure, kpathtype='vasp')

    assert bandjdata['task_options']['kline_type'] == 'vasp'
    assert bandjdata['task_options']['task'] == 'band'
    assert isinstance(bandjdata['task_options']['kpath'],list) or isinstance(bandjdata['task_options']['kpath'],str)
    assert isinstance(bandjdata['task_options']['high_sym_kpoints'],dict)
    assert isinstance(bandjdata['task_options']['number_in_line'],int)

    assert isinstance(common_options['basis'], dict)

    assert len(common_options['basis']) == 2
    assert 'Mo' in common_options['basis']
    assert 'S' in common_options['basis']
