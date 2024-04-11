#test_negf_Device_set_leadLR
from dptb.negf.device_property import DeviceProperty
from dptb.nn.build import build_model
import json
from dptb.utils.make_kpoints import kmesh_sampling
from dptb.utils.tools import j_must_have
from dptb.utils.tools import j_loader
import numpy as np
import torch
from dptb.negf.negf_hamiltonian_init import NEGFHamiltonianInit
from ase.io import read
from dptb.negf.lead_property import LeadProperty
from dptb.utils.constants import Boltzmann, eV2J
import pytest


@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    """
    :return:
    """
    return str(request.config.rootdir)

def test_negf_Device(root_directory):
    model_ckpt=root_directory +'/dptb/tests/data/test_negf/test_negf_run/nnsk_C.json'
    results_path=root_directory +"/dptb/tests/data/test_negf"
    input_path = root_directory +"/dptb/tests/data/test_negf/test_negf_hamiltonian/run_input.json"
    structure=root_directory +"/dptb/tests/data/test_negf/test_negf_run/chain.vasp"
    # log_path=root_directory +"/dptb/tests/data/test_negf/test_negf_Device/test.log"
    negf_json = json.load(open(input_path))
    model = build_model(model_ckpt,model_options=negf_json['model_options'],common_options=negf_json['common_options'])


    hamiltonian = NEGFHamiltonianInit(model=model,
                                    AtomicData_options=negf_json['AtomicData_options'], 
                                    structure=structure,
                                    pbc_negf = negf_json['task_options']["stru_options"]['pbc'], 
                                    stru_options = negf_json['task_options']['stru_options'],
                                    unit = negf_json['task_options']['unit'], 
                                    results_path=results_path,
                                    torch_device = torch.device('cpu'))

    # hamiltonian = NEGFHamiltonianInit(apiH=apiHrk, structase=structase, stru_options=task_options["stru_options"], results_path=results_path)
    kpoints= kmesh_sampling(negf_json['task_options']["stru_options"]["kmesh"])
    with torch.no_grad():
        struct_device, struct_leads = hamiltonian.initialize(kpoints=kpoints)

    
    deviceprop = DeviceProperty(hamiltonian, struct_device, results_path=results_path, efermi=negf_json['task_options']['e_fermi'])
    deviceprop.set_leadLR(
            lead_L=LeadProperty(
            hamiltonian=hamiltonian, 
            tab="lead_L", 
            structure=struct_leads["lead_L"], 
            results_path=results_path,
            e_T=negf_json['task_options']['ele_T'],
            efermi=negf_json['task_options']['e_fermi'], 
            voltage=negf_json['task_options']["stru_options"]["lead_L"]["voltage"]
        ),
            lead_R=LeadProperty(
            hamiltonian=hamiltonian, 
            tab="lead_R", 
            structure=struct_leads["lead_R"], 
            results_path=results_path,
            e_T=negf_json['task_options']['ele_T'],
            efermi=negf_json['task_options']['e_fermi'], 
            voltage=negf_json['task_options']["stru_options"]["lead_R"]["voltage"]
        )
    )
    
    # check device.Lead_L.structure
    assert all(deviceprop.lead_L.structure.symbols=='C4')
    assert deviceprop.lead_L.structure.pbc[0]==False
    assert deviceprop.lead_L.structure.pbc[1]==False
    assert deviceprop.lead_L.structure.pbc[2]==True
    assert np.diag(np.array((deviceprop.lead_L.structure.cell-[10.0, 10.0, 6.4])<1e-4)).all()
    assert deviceprop.lead_L.tab=="lead_L"
    assert abs(deviceprop.mu+13.638587951660156)<1e-5
    # check device.Lead_R.structure
    assert all(deviceprop.lead_R.structure.symbols=='C4')
    assert deviceprop.lead_R.structure.pbc[0]==False
    assert deviceprop.lead_R.structure.pbc[1]==False
    assert deviceprop.lead_R.structure.pbc[2]==True
    assert np.diag(np.array((deviceprop.lead_R.structure.cell-[10.0, 10.0, 6.4])<1e-4)).all()
    assert deviceprop.lead_R.tab=="lead_R"


    # calculate Self energy and Green function
    stru_options = j_must_have(negf_json['task_options'], "stru_options")
    leads = stru_options.keys()
    for ll in leads:
        if ll.startswith("lead"): #calculate surface green function at E=0
            getattr(deviceprop, ll).self_energy(
                energy=torch.tensor([0]), 
                kpoint=kpoints[0], 
                eta_lead=negf_json['task_options']["eta_lead"],
                method=negf_json['task_options']["sgf_solver"]
                )

        # check left and right leads' self-energy
    lead_L_se_standard=torch.tensor([[0.0000e+00-0.61690134j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],
        [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],
        [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],
        [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j]], dtype=torch.complex128)
    lead_R_se_standard=torch.tensor([[ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],
        [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],
        [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],
        [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00-0.51628502j]], dtype=torch.complex128)

    
    assert  abs(deviceprop.lead_L.se-lead_L_se_standard).max()<1e-5
    assert  abs(deviceprop.lead_R.se-lead_R_se_standard).max()<1e-5

    deviceprop.cal_green_function(  energy=torch.tensor([0]),   #calculate device green function at E=0
                            kpoint=kpoints[0], 
                            eta_device=negf_json['task_options']["eta_device"], 
                            block_tridiagonal=negf_json['task_options']["block_tridiagonal"]
                            )

    #check  green functions' results
    assert list(deviceprop.greenfuncs.keys())==['g_trans', 'grd', 'grl', 'gru', 'gr_left', 'gnd', 'gnl',\
                                        'gnu', 'gin_left', 'gpd', 'gpl', 'gpu', 'gip_left']
    g_trans= torch.tensor([[ 0.00000000-0.74812411j, -0.88334131-0.00000000j,
          0.00000000+0.74812499j,  0.88333889+0.00000000j],
        [-0.88334131-0.00000000j,  0.00000000-0.89392734j,
         -0.75709057-0.00000000j, -0.00000000+0.89392489j],
        [-0.00000000+0.74812499j, -0.75709057-0.00000000j,
          0.00000000-0.74812586j, -0.88333993-0.00000000j],
        [ 0.88333889+0.00000000j,  0.00000000+0.89392489j,
         -0.88333993+0.00000000j, -0.00000000-0.89392244j]],dtype=torch.complex128)
    
    grd= [torch.tensor([[ 0.00000000-0.74812411j, -0.88334131-0.00000000j,
          0.00000000+0.74812499j,  0.88333889+0.00000000j],
        [-0.88334131-0.00000000j,  0.00000000-0.89392734j,
         -0.75709057-0.00000000j, -0.00000000+0.89392489j],
        [-0.00000000+0.74812499j, -0.75709057-0.00000000j,
          0.00000000-0.74812586j, -0.88333993-0.00000000j],
        [ 0.88333889+0.00000000j,  0.00000000+0.89392489j,
         -0.88333993+0.00000000j, -0.00000000-0.89392244j]],dtype=torch.complex128)]

    assert  abs(g_trans-deviceprop.greenfuncs['g_trans']).max()<1e-5
    assert  abs(grd[0]-deviceprop.greenfuncs['grd'][0]).max()<1e-5
    assert deviceprop.greenfuncs['grl'] == []
    assert deviceprop.greenfuncs['gru'] == []

    gr_left= [torch.tensor([[ 0.00000000-0.74812411j, -0.88334131-0.00000000j,
          0.00000000+0.74812499j,  0.88333889+0.00000000j],
        [-0.88334131-0.00000000j,  0.00000000-0.89392734j,
         -0.75709057-0.00000000j, -0.00000000+0.89392489j],
        [-0.00000000+0.74812499j, -0.75709057-0.00000000j,
          0.00000000-0.74812586j, -0.88333993-0.00000000j],
        [ 0.88333889+0.00000000j,  0.00000000+0.89392489j,
         -0.88333993+0.00000000j, -0.00000000-0.89392244j]],
            dtype=torch.complex128)]

    gnd = [torch.tensor([[ 0.74812411+0.00000000e+00j,  0.00000000-5.55111512e-17j,
         -0.74812499+0.00000000e+00j,  0.00000000+5.55111512e-17j],
        [ 0.00000000+1.11022302e-16j,  0.89392734+0.00000000e+00j,
          0.00000000+5.55111512e-17j, -0.89392489+0.00000000e+00j],
        [-0.74812499+0.00000000e+00j,  0.00000000-1.11022302e-16j,
          0.74812586+0.00000000e+00j,  0.00000000+1.11022302e-16j],
        [ 0.00000000-5.55111512e-17j, -0.89392489+0.00000000e+00j,
          0.00000000-1.11022302e-16j,  0.89392244+0.00000000e+00j]],dtype=torch.complex128)]

    assert  abs(gr_left[0]-deviceprop.greenfuncs['gr_left'][0]).max()<1e-5
    assert  abs(gnd[0]-deviceprop.greenfuncs['gnd'][0]).max()<1e-5
    assert deviceprop.greenfuncs['gnl'] == []
    assert deviceprop.greenfuncs['gnu'] == []

    gin_left=[torch.tensor([[ 0.74812411+0.00000000e+00j,  0.00000000-5.55111512e-17j,
         -0.74812499+0.00000000e+00j,  0.00000000+5.55111512e-17j],
        [ 0.00000000+1.11022302e-16j,  0.89392734+0.00000000e+00j,
          0.00000000+5.55111512e-17j, -0.89392489+0.00000000e+00j],
        [-0.74812499+0.00000000e+00j,  0.00000000-1.11022302e-16j,
          0.74812586+0.00000000e+00j,  0.00000000+1.11022302e-16j],
        [ 0.00000000-5.55111512e-17j, -0.89392489+0.00000000e+00j,
          0.00000000-1.11022302e-16j,  0.89392244+0.00000000e+00j]],dtype=torch.complex128)]


    assert  abs(gin_left[0]-deviceprop.greenfuncs['gin_left'][0]).max()<1e-5

    assert deviceprop.greenfuncs['gpd']== None
    assert deviceprop.greenfuncs['gpl']== None
    assert deviceprop.greenfuncs['gpu']== None
    assert deviceprop.greenfuncs['gip_left']== None

    Tc=deviceprop._cal_tc_() #transmission
    assert abs(Tc-1)<1e-2

    dos = deviceprop._cal_dos_()
    dos_standard = torch.tensor(2.090723, dtype=torch.float64)
    assert abs(dos-dos_standard)<1e-4

    ldos = deviceprop._cal_ldos_()
    torch.set_printoptions(precision=6)
    print(ldos)
    ldos_standard = torch.tensor([0.2611, 0.2611, 0.2611, 0.2611], dtype=torch.float64)*2
    assert abs(ldos_standard-ldos).max()<1e-4




