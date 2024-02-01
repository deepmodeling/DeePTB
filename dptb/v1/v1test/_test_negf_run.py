from dptb.entrypoints import run
import pytest
import torch


@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    """
    :return:
    """
    return str(request.config.rootdir)

# NEGF calculaion in 1D carbon chain with zero-bias transmission 1 G0

def test_negf_run(root_directory):
    INPUT_file =  root_directory +"/dptb/tests/data/test_negf/test_negf_run/input_negf.json" 
    output =  root_directory +"/dptb/tests/data/test_negf/test_negf_run/out_negf"  
    checkfile =  root_directory +'/dptb/tests/data/test_negf/test_negf_run/nnsk_C.json'
    structure =  root_directory +"/dptb/tests/data/test_negf/test_negf_run/chain.vasp" 

    run(INPUT=INPUT_file,init_model=checkfile,output=output,run_sk=True,structure=structure,\
    log_level=5,log_path=output+"/test.log",use_correction=False)

    negf_results = torch.load(output+"/results/negf.k0.out.pth")
    trans = negf_results['TC']
    assert(abs(trans[int(len(trans)/2)]-1)<1e-5)  #compare with calculated transmission at efermi




