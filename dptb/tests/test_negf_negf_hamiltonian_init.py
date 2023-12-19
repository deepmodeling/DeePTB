# Hamiltonian
from dptb.v1.init_nnsk import InitSKModel
from dptb.nnops.v1.NN2HRK import NN2HRK
from dptb.nnops.v1.apihost import NNSKHost
from dptb.utils.tools import j_must_have
from dptb.utils.tools import j_loader
import numpy as np
import torch
import pytest
from dptb.negf.negf_hamiltonian_init import NEGFHamiltonianInit
from ase.io import read


@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    """
    :return:
    """
    return str(request.config.rootdir)


def test_negf_Hamiltonian(root_directory):

    model_ckpt=root_directory +'/dptb/tests/data/test_negf/test_negf_run/nnsk_C.json'
    jdata = root_directory +"/dptb/tests/data/test_negf/test_negf_hamiltonian/run_input.json"
    structure=root_directory +"/dptb/tests/data/test_negf/test_negf_run/chain.vasp"
    log_path=root_directory +"/dptb/tests/data/test_negf/test_negf_hamiltonian/test.log"

    apihost = NNSKHost(checkpoint=model_ckpt, config=jdata)
    apihost.register_plugin(InitSKModel())
    apihost.build()
    apiHrk = NN2HRK(apihost=apihost, mode='nnsk')
    jdata = j_loader(jdata)
    task_options = j_must_have(jdata, "task_options")

    run_opt = {
            "run_sk": True,
            "init_model":model_ckpt,
            "results_path":root_directory +"/dptb/tests/data/test_negf/test_negf_hamiltonian/",
            "structure":structure,
            "log_path": log_path,
            "log_level": 5,
            "use_correction":False
        }


    structase=read(run_opt['structure'])
    results_path=run_opt.get('results_path')
    kpoints=np.array([[0,0,0]])

    hamiltonian = NEGFHamiltonianInit(apiH=apiHrk, structase=structase, stru_options=task_options["stru_options"], results_path=results_path)
    with torch.no_grad():
        struct_device, struct_leads = hamiltonian.initialize(kpoints=kpoints)
    
    #check device's Hamiltonian
    assert all(struct_device.symbols=="C4")
    assert all(struct_device.pbc)==False
    assert np.diag(np.array(struct_device.cell==[10.0, 10.0, 19.2])).all()
    
    #check lead_L's Hamiltonian
 
    assert all(struct_leads["lead_L"].symbols=="C4")
    assert struct_leads["lead_L"].pbc[0]==False
    assert struct_leads["lead_L"].pbc[1]==False
    assert struct_leads["lead_L"].pbc[2]==True
    assert np.diag(np.array(struct_leads["lead_L"].cell==[10.0, 10.0, -6.4])).all()
    
    #check lead_R's Hamiltonian
 
    assert all(struct_leads["lead_R"].symbols=="C4")
    assert struct_leads["lead_R"].pbc[0]==False
    assert struct_leads["lead_R"].pbc[1]==False
    assert struct_leads["lead_R"].pbc[2]==True
    assert np.diag(np.array(struct_leads["lead_R"].cell==[10.0, 10.0, 6.4])).all()


    #check hs_device
    h_device = hamiltonian.get_hs_device(kpoint=np.array([0,0,0]),V=0,block_tridiagonal=False)[0][0]
    print(hamiltonian.get_hs_device(kpoint=np.array([0,0,0]),V=0,block_tridiagonal=False)[0])
    h_device_standard = torch.tensor([[-13.6386+0.j,   0.6096+0.j,   0.0000+0.j,   0.0000+0.j],
        [  0.6096+0.j, -13.6386+0.j,   0.6096+0.j,   0.0000+0.j],
        [  0.0000+0.j,   0.6096+0.j, -13.6386+0.j,   0.6096+0.j],
        [  0.0000+0.j,   0.0000+0.j,   0.6096+0.j, -13.6386+0.j]],dtype=torch.complex128)
    assert abs(h_device-h_device_standard).max()<1e-4

    s_device = hamiltonian.get_hs_device(kpoint=np.array([0,0,0]),V=0,block_tridiagonal=False)[1][0]
    print(hamiltonian.get_hs_device(kpoint=np.array([0,0,0]),V=0,block_tridiagonal=False)[1][0])
    s_standard = torch.eye(4)
    assert abs(s_device-s_standard).max()<1e-5



    #check hs_lead
    hl_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[0][0]
    hl_lead_standard = torch.tensor([-13.6386+0.j,   0.6096+0.j], dtype=torch.complex128)    
    assert abs(hl_lead-hl_lead_standard).max()<1e-4

    hll_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[1][0]
    hll_lead_standard = torch.tensor([0.0000+0.j, 0.6096+0.j], dtype=torch.complex128)
    print(hll_lead)    
    assert abs(hll_lead-hll_lead_standard).max()<1e-4

    hDL_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[2]
    hDL_lead_standard = torch.tensor([[0.0000+0.j, 0.6096+0.j],
        [0.0000+0.j, 0.0000+0.j],
        [0.0000+0.j, 0.0000+0.j],
        [0.0000+0.j, 0.0000+0.j]], dtype=torch.complex128)   
    assert abs(hDL_lead-hDL_lead_standard).max()<1e-5

    sl_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[3]
    sl_lead_standard = torch.eye(2)   
    assert abs(sl_lead-sl_lead_standard).max()<1e-5

    sll_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[4]
    sll_lead_standard = torch.zeros(2)    
    assert abs(sll_lead-sll_lead_standard).max()<1e-5

    sDL_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[5]
    sDL_lead_standard = torch.zeros([4,2])    
    assert abs(sDL_lead-sDL_lead_standard).max()<1e-5   


    # check device norbs
    na = len(hamiltonian.device_norbs)
    device_norbs_standard=[1,1,1,1]
    assert na == 4
    assert hamiltonian.device_norbs==device_norbs_standard

   
