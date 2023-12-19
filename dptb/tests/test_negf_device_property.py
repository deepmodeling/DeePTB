#test_negf_Device_set_leadLR
from dptb.negf.device_property import DeviceProperty
from dptb.v1.init_nnsk import InitSKModel
from dptb.nnops.v1.NN2HRK import NN2HRK
from dptb.nnops.v1.apihost import NNSKHost
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
    jdata = root_directory +"/dptb/tests/data/test_negf/test_negf_hamiltonian/run_input.json"
    structure=root_directory +"/dptb/tests/data/test_negf/test_negf_run/chain.vasp"
    log_path=root_directory +"/dptb/tests/data/test_negf/test_negf_Device/test.log"


    # read input files and generate Hamiltonian
    apihost = NNSKHost(checkpoint=model_ckpt, config=jdata)
    apihost.register_plugin(InitSKModel())
    apihost.build()
    apiHrk = NN2HRK(apihost=apihost, mode='nnsk')
    jdata = j_loader(jdata)
    task_options = j_must_have(jdata, "task_options")

    run_opt = {
            "run_sk": True,
            "init_model":model_ckpt,
            "results_path":root_directory +"/dptb/tests/data/test_negf/",
            "structure":structure,
            "log_path": log_path,
            "log_level": 5,
            "use_correction":False
        }


    structase=read(run_opt['structure'])
    results_path=run_opt.get('results_path')
    kpoints=np.array([[0,0,0]])
    ele_T = task_options["ele_T"]
    kBT = Boltzmann * ele_T / eV2J
    e_fermi = task_options["e_fermi"]

    hamiltonian = NEGFHamiltonianInit(apiH=apiHrk, structase=structase, stru_options=task_options["stru_options"], results_path=results_path)
    with torch.no_grad():
        struct_device, struct_leads = hamiltonian.initialize(kpoints=kpoints)


    device = DeviceProperty(hamiltonian, struct_device, results_path=results_path, efermi=e_fermi)
    device.set_leadLR(
                    lead_L=LeadProperty(
                    hamiltonian=hamiltonian, 
                    tab="lead_L", 
                    structure=struct_leads["lead_L"], 
                    results_path=results_path,
                    e_T=ele_T,
                    efermi=e_fermi, 
                    voltage=task_options["stru_options"]["lead_L"]["voltage"]
                ),
                    lead_R=LeadProperty(
                        hamiltonian=hamiltonian, 
                        tab="lead_R", 
                        structure=struct_leads["lead_R"], 
                        results_path=results_path, 
                        e_T=ele_T,
                        efermi=e_fermi, 
                        voltage=task_options["stru_options"]["lead_R"]["voltage"]
                )
            )
    
    # check device.Lead_L.structure
    assert all(device.lead_L.structure.symbols=='C4')
    assert device.lead_L.structure.pbc[0]==False
    assert device.lead_L.structure.pbc[1]==False
    assert device.lead_L.structure.pbc[2]==True
    assert np.diag(np.array((device.lead_L.structure.cell-[10.0, 10.0, 6.4])<1e-4)).all()
    assert device.lead_L.tab=="lead_L"
    assert abs(device.mu+13.638587951660156)<1e-5
    # check device.Lead_R.structure
    assert all(device.lead_R.structure.symbols=='C4')
    assert device.lead_R.structure.pbc[0]==False
    assert device.lead_R.structure.pbc[1]==False
    assert device.lead_R.structure.pbc[2]==True
    assert np.diag(np.array((device.lead_R.structure.cell-[10.0, 10.0, 6.4])<1e-4)).all()
    assert device.lead_R.tab=="lead_R"


    # calculate Self energy and Green function
    stru_options = j_must_have(task_options, "stru_options")
    leads = stru_options.keys()
    for ll in leads:
        if ll.startswith("lead"): #calculate surface green function at E=0
            getattr(device, ll).self_energy(
                energy=torch.tensor([0]), 
                kpoint=kpoints[0], 
                eta_lead=task_options["eta_lead"],
                method=task_options["sgf_solver"]
                )

        # check left and right leads' self-energy
    lead_L_se_standard=torch.tensor([[-3.3171e-07-0.6096j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],
        [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],
        [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],
        [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j]], dtype=torch.complex128)
    lead_R_se_standard=torch.tensor([[ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],
        [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],
        [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],
        [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,-3.3171e-07-0.6096j]], dtype=torch.complex128)

    assert  abs(device.lead_L.se-lead_L_se_standard).max()<1e-5
    assert  abs(device.lead_R.se-lead_R_se_standard).max()<1e-5

    device.cal_green_function(  energy=torch.tensor([0]),   #calculate device green function at E=0
                            kpoint=kpoints[0], 
                            eta_device=task_options["eta_device"], 
                            block_tridiagonal=task_options["block_tridiagonal"]
                            )

    #check  green functions' results
    assert list(device.greenfuncs.keys())==['g_trans', 'grd', 'grl', 'gru', 'gr_left', 'gnd', 'gnl',\
                                        'gnu', 'gin_left', 'gpd', 'gpl', 'gpu', 'gip_left']
    g_trans= torch.tensor([[ 1.0983e-11-8.2022e-01j, -8.2022e-01+4.4634e-07j,8.9264e-07+8.2022e-01j,  8.2022e-01-1.3390e-06j],
            [-8.2022e-01+4.4634e-07j, -3.6607e-12-8.2022e-01j,-8.2021e-01+4.4631e-07j,  8.9264e-07+8.2022e-01j],
            [ 8.9264e-07+8.2022e-01j, -8.2021e-01+4.4631e-07j,-3.6607e-12-8.2022e-01j, -8.2022e-01+4.4634e-07j],
            [ 8.2022e-01-1.3390e-06j,  8.9264e-07+8.2022e-01j,-8.2022e-01+4.4634e-07j,  1.0983e-11-8.2022e-01j]],dtype=torch.complex128)
    grd= [torch.tensor([[ 1.0983e-11-8.2022e-01j, -8.2022e-01+4.4634e-07j,8.9264e-07+8.2022e-01j,  8.2022e-01-1.3390e-06j],
            [-8.2022e-01+4.4634e-07j, -3.6607e-12-8.2022e-01j,-8.2021e-01+4.4631e-07j,  8.9264e-07+8.2022e-01j],
            [ 8.9264e-07+8.2022e-01j, -8.2021e-01+4.4631e-07j,-3.6607e-12-8.2022e-01j, -8.2022e-01+4.4634e-07j],
            [ 8.2022e-01-1.3390e-06j,  8.9264e-07+8.2022e-01j,-8.2022e-01+4.4634e-07j,  1.0983e-11-8.2022e-01j]],dtype=torch.complex128)]

    assert  abs(g_trans-device.greenfuncs['g_trans']).max()<1e-5
    assert  abs(grd[0]-device.greenfuncs['grd'][0]).max()<1e-5
    assert device.greenfuncs['grl'] == []
    assert device.greenfuncs['gru'] == []

    gr_left= [torch.tensor([[ 1.0983e-11-8.2022e-01j, -8.2022e-01+4.4634e-07j,8.9264e-07+8.2022e-01j,  8.2022e-01-1.3390e-06j],
            [-8.2022e-01+4.4634e-07j, -3.6607e-12-8.2022e-01j,-8.2021e-01+4.4631e-07j,  8.9264e-07+8.2022e-01j],
            [ 8.9264e-07+8.2022e-01j, -8.2021e-01+4.4631e-07j,-3.6607e-12-8.2022e-01j, -8.2022e-01+4.4634e-07j],
            [ 8.2022e-01-1.3390e-06j,  8.9264e-07+8.2022e-01j,-8.2022e-01+4.4634e-07j,  1.0983e-11-8.2022e-01j]],dtype=torch.complex128)]

    gnd = [torch.tensor([[ 8.2022e-01+0.0000e+00j, -4.4634e-07+2.2204e-16j,-8.2022e-01-3.1764e-22j,  1.3390e-06-5.5511e-17j],
        [-4.4634e-07-2.7756e-16j,  8.2022e-01+2.6470e-23j, -4.4631e-07+2.7756e-16j, -8.2022e-01-2.3823e-22j],
        [-8.2022e-01+2.9117e-22j, -4.4631e-07-2.2204e-16j, 8.2022e-01+7.9409e-23j, -4.4634e-07+1.1102e-16j],
        [ 1.3390e-06+5.5511e-17j, -8.2022e-01+2.1176e-22j, -4.4634e-07-1.1102e-16j,  8.2022e-01+0.0000e+00j]],dtype=torch.complex128)]

    assert  abs(gr_left[0]-device.greenfuncs['gr_left'][0]).max()<1e-5
    assert  abs(gnd[0]-device.greenfuncs['gnd'][0]).max()<1e-5
    assert device.greenfuncs['gnl'] == []
    assert device.greenfuncs['gnu'] == []

    gin_left=[torch.tensor([[ 8.2022e-01+0.0000e+00j, -4.4634e-07+2.2204e-16j, -8.2022e-01-3.1764e-22j,  1.3390e-06-5.5511e-17j],
        [-4.4634e-07-2.7756e-16j,  8.2022e-01+2.6470e-23j, -4.4631e-07+2.7756e-16j, -8.2022e-01-2.3823e-22j],
        [-8.2022e-01+2.9117e-22j, -4.4631e-07-2.2204e-16j, 8.2022e-01+7.9409e-23j, -4.4634e-07+1.1102e-16j],
        [ 1.3390e-06+5.5511e-17j, -8.2022e-01+2.1176e-22j,-4.4634e-07-1.1102e-16j,  8.2022e-01+0.0000e+00j]],dtype=torch.complex128)]
    assert  abs(gin_left[0]-device.greenfuncs['gin_left'][0]).max()<1e-5

    assert device.greenfuncs['gpd']== None
    assert device.greenfuncs['gpl']== None
    assert device.greenfuncs['gpu']== None
    assert device.greenfuncs['gip_left']== None

    Tc=device._cal_tc_() #transmission
    assert abs(Tc-1)<1e-5

    dos = device._cal_dos_()
    dos_standard = torch.tensor(2.0887, dtype=torch.float64)
    assert abs(dos-dos_standard)<1e-4

    ldos = device._cal_ldos_()
    ldos_standard = torch.tensor([0.2611, 0.2611, 0.2611, 0.2611], dtype=torch.float64)*2
    assert abs(ldos_standard-ldos).max()<1e-4




