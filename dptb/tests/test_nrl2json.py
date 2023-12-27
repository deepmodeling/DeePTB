import pytest
from dptb.utils.read_NRL_tojson import read_nrl_file,  nrl2dptb
import json
import numpy as np

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
        return str(request.config.rootdir)


def test_nrl2json_v0(root_directory):
    nrl_file = root_directory + '/dptb/tests/data/nrl/Cu.par'
    ref_nrl_file = root_directory + '/dptb/tests/data/nrl/nrl_Cu_ckpt.json'
    input = root_directory + '/dptb/tests/data/nrl/input_Cu.json'

    with open(ref_nrl_file,'r') as f:
        ref_nrl_dict = json.load(f)

    NRL_data = read_nrl_file(nrl_file)
    input_dict, nrl_tb_dict =  nrl2dptb(input, NRL_data)

    # check input_dict and  ref_input_dict are the same, with the same keys and values.
    assert input_dict['common_options']['unit'] == 'Ry'
    assert abs(input_dict['common_options']['onsite_cutoff'] - 8.7314239798995) < 1E-8
    assert input_dict['common_options']['overlap'] == True
    assert input_dict['model_options']['skfunction']['skformula'] == "NRLv0"
    assert abs(input_dict['model_options']['skfunction']['sk_cutoff'] - 8.7314239798995) < 1E-8
    assert abs(input_dict['model_options']['skfunction']['sk_decay_w'] - 0.26459) < 1E-4
    assert abs(input_dict['model_options']['onsitefuncion']['onsite_func_cutoff'] - 8.7314239798995) < 1E-8
    assert abs(input_dict['model_options']['onsitefuncion']['onsite_func_decay_w'] - 0.26459) < 1E-4
    assert abs(input_dict['model_options']['onsitefuncion']['onsite_func_lambda'] - 2.024780663957271) < 1E-8

    # check nrl_tb_dict and  ref_nrl_dict are the same, with the same keys and values.

    assert set(list(nrl_tb_dict.keys())) == set(list(ref_nrl_dict.keys()))
    for ikey in nrl_tb_dict.keys():
        sub_dict = nrl_tb_dict[ikey]
        sub_ref_dict = ref_nrl_dict[ikey]
        assert set(list(sub_dict.keys())) == set(list(sub_ref_dict.keys()))
        for jkey in sub_dict.keys():
            assert (np.abs(np.asarray(sub_dict[jkey]) - np.asarray(sub_ref_dict[jkey])) < 1E-6).all()


def test_nrl2json_v1_sp(root_directory):
    nrl_file = root_directory + '/dptb/tests/data/nrl/Si_sp.par'
    ref_nrl_file = root_directory + '/dptb/tests/data/nrl/nrl_Si_sp_ckpt.json'
    input = root_directory + '/dptb/tests/data/nrl/input_Si_sp.json'

    with open(ref_nrl_file,'r') as f:
        ref_nrl_dict = json.load(f)

    NRL_data = read_nrl_file(nrl_file)
    input_dict, nrl_tb_dict =  nrl2dptb(input, NRL_data)

    # check input_dict and  ref_input_dict are the same, with the same keys and values.
    assert input_dict['common_options']['unit'] == 'Ry'
    assert abs(input_dict['common_options']['onsite_cutoff'] - 6.6147151362875) < 1E-8
    assert input_dict['common_options']['overlap'] == True
    assert input_dict['model_options']['skfunction']['skformula'] == "NRLv1"
    assert abs(input_dict['model_options']['skfunction']['sk_cutoff'] - 6.6147151362875) < 1E-8
    assert abs(input_dict['model_options']['skfunction']['sk_decay_w'] - 0.2645886054515) < 1E-4
    assert abs(input_dict['model_options']['onsitefuncion']['onsite_func_cutoff'] - 6.6147151362875) < 1E-8
    assert abs(input_dict['model_options']['onsitefuncion']['onsite_func_decay_w'] - 0.2645886054515) < 1E-4
    assert abs(input_dict['model_options']['onsitefuncion']['onsite_func_lambda'] - 1.517042837140912) < 1E-8

    # check nrl_tb_dict and  ref_nrl_dict are the same, with the same keys and values.

    assert set(list(nrl_tb_dict.keys())) == set(list(ref_nrl_dict.keys()))
    for ikey in nrl_tb_dict.keys():
        sub_dict = nrl_tb_dict[ikey]
        sub_ref_dict = ref_nrl_dict[ikey]
        assert set(list(sub_dict.keys())) == set(list(sub_ref_dict.keys()))
        for jkey in sub_dict.keys():
            assert (np.abs(np.asarray(sub_dict[jkey]) - np.asarray(sub_ref_dict[jkey])) < 1E-6).all()



def test_nrl2json_v1_spd(root_directory):
    nrl_file = root_directory + '/dptb/tests/data/nrl/Si_spd.par'
    ref_nrl_file = root_directory + '/dptb/tests/data/nrl/nrl_Si_spd_ckpt.json'
    input = root_directory + '/dptb/tests/data/nrl/input_Si_spd.json'

    with open(ref_nrl_file,'r') as f:
        ref_nrl_dict = json.load(f)

    NRL_data = read_nrl_file(nrl_file)
    input_dict, nrl_tb_dict =  nrl2dptb(input, NRL_data)

    # check input_dict and  ref_input_dict are the same, with the same keys and values.
    assert input_dict['common_options']['unit'] == 'Ry'
    assert abs(input_dict['common_options']['onsite_cutoff'] - 6.6147151362875) < 1E-8
    assert input_dict['common_options']['overlap'] == True
    assert input_dict['model_options']['skfunction']['skformula'] == "NRLv1"
    assert abs(input_dict['model_options']['skfunction']['sk_cutoff'] - 6.6147151362875) < 1E-8
    assert abs(input_dict['model_options']['skfunction']['sk_decay_w'] - 0.2645886054515) < 1E-4
    assert abs(input_dict['model_options']['onsitefuncion']['onsite_func_cutoff'] - 6.6147151362875) < 1E-8
    assert abs(input_dict['model_options']['onsitefuncion']['onsite_func_decay_w'] - 0.2645886054515) < 1E-4
    assert abs(input_dict['model_options']['onsitefuncion']['onsite_func_lambda'] - 1.5269575694188455) < 1E-8

    # check nrl_tb_dict and  ref_nrl_dict are the same, with the same keys and values.

    assert set(list(nrl_tb_dict.keys())) == set(list(ref_nrl_dict.keys()))
    for ikey in nrl_tb_dict.keys():
        sub_dict = nrl_tb_dict[ikey]
        sub_ref_dict = ref_nrl_dict[ikey]
        assert set(list(sub_dict.keys())) == set(list(sub_ref_dict.keys()))
        for jkey in sub_dict.keys():
            assert (np.abs(np.asarray(sub_dict[jkey]) - np.asarray(sub_ref_dict[jkey])) < 1E-6).all()