#test_negf_RGF
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
from dptb.utils.make_kpoints import kmesh_sampling
from dptb.negf.lead_property import LeadProperty
from dptb.utils.constants import Boltzmann, eV2J
import os
from dptb.negf.recursive_green_cal import recursive_gf
import pytest

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    """
    :return:
    """
    return str(request.config.rootdir)

def test_negf_RGF(root_directory):
    model_ckpt=root_directory +'/dptb/tests/data/test_negf/test_negf_run/nnsk_C.json'
    jdata = root_directory +"/dptb/tests/data/test_negf/test_negf_hamiltonian/run_input.json"
    structure=root_directory +"/dptb/tests/data/test_negf/test_negf_run/chain.vasp"
    log_path=root_directory +"/dptb/tests/data/test_negf/test_negf_Device/test.log"

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
    kpoints= kmesh_sampling(task_options["stru_options"]["kmesh"])
    ele_T = task_options["ele_T"]
    kBT = Boltzmann * ele_T / eV2J
    e_fermi = task_options["e_fermi"]

    hamiltonian = NEGFHamiltonianInit(apiH=apiHrk, structase=structase, stru_options=task_options["stru_options"], results_path=results_path)
    with torch.no_grad():
        struct_device, struct_leads = hamiltonian.initialize(kpoints=kpoints)

    #initial class Device 
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

    # green function part
    e=torch.tensor([0])
    kpoint=kpoints[0];eta_device=task_options["eta_device"];block_tridiagonal=task_options["block_tridiagonal"]

    assert len(np.array(kpoint).reshape(-1)) == 3
    if not isinstance(e, torch.Tensor):
        e = torch.tensor(e, dtype=torch.complex128)

    if os.path.exists(os.path.join(results_path, "POTENTIAL.pth")):
        V = torch.load(os.path.join(results_path, "POTENTIAL.pth"), weights_only=False)
    elif abs(device.mu - device.efermi) < 1e-7:
        V = device.efermi - device.mu
    else:
        V = 0.

    if not hasattr(device, "hd") or not hasattr(device, "sd"):
        device.hd, device.sd, _, _, _, _ = hamiltonian.get_hs_device(kpoint,V, block_tridiagonal)
    s_in = [torch.zeros(i.shape).cdouble() for i in device.hd]

    tags = ["g_trans","grd", "grl", "gru", "gr_left", \
            "gnd", "gnl", "gnu", "gin_left", \
            "gpd", "gpl", "gpu", "gip_left"]

    #generate recursive_gf input
    seL = device.lead_L.se #self energy
    seR = device.lead_R.se
    seinL = seL * device.lead_L.fermi_dirac(e+device.mu).reshape(-1)
    seinR = seR * device.lead_R.fermi_dirac(e+device.mu).reshape(-1)

    seL_standard = torch.tensor([[-3.3171e-07-0.6096j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],\
            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],\
            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],\
            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j]], dtype=torch.complex128)
    seR_standard = torch.tensor([[ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],\
            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j, 0.0000e+00+0.0000j],\
                            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],\
            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,-3.3171e-07-0.6096j]], dtype=torch.complex128)
    seinL_standard = torch.tensor([[-1.6585e-07-0.3048j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],\
            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],\
            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],\
            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j]], dtype=torch.complex128)
    seinR_standard = torch.tensor([[ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],\
            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],\
            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],\
            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,-1.6585e-07-0.3048j]], dtype=torch.complex128)

    assert abs(seL-seL_standard).max()<1e-5
    assert abs(seR-seR_standard).max()<1e-5
    assert abs(seinL-seinL_standard).max()<1e-5
    assert abs(seinR-seinR_standard).max()<1e-5



    s01, s02 = s_in[0].shape
    se01, se02 = seL.shape
    idx0, idy0 = min(s01, se01), min(s02, se02)

    s11, s12 = s_in[-1].shape
    se11, se12 = seR.shape
    idx1, idy1 = min(s11, se11), min(s12, se12)

    s_in[0][:idx0,:idy0] = s_in[0][:idx0,:idy0] + seinL[:idx0,:idy0]
    s_in[-1][-idx1:,-idy1:] = s_in[-1][-idx1:,-idy1:] + seinR[-idx1:,-idy1:]

    s_in0_standard = torch.tensor([[-1.6585e-07-0.3048j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],\
            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j, 0.0000e+00+0.0000j],\
            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],\
            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,-1.6585e-07-0.3048j]], dtype=torch.complex128)
    s_in_1_standard = torch.tensor([[-1.6585e-07-0.3048j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,0.0000e+00+0.0000j],\
            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j, 0.0000e+00+0.0000j],\
            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j, 0.0000e+00+0.0000j],\
            [ 0.0000e+00+0.0000j,  0.0000e+00+0.0000j,  0.0000e+00+0.0000j,-1.6585e-07-0.3048j]], dtype=torch.complex128)
    assert abs(s_in[0]-s_in0_standard).max()<1e-5
    assert abs(s_in[-1]-s_in_1_standard).max()<1e-5


    ans = recursive_gf(e, hl=[], hd=device.hd, hu=[],
                        sd=device.sd, su=[], sl=[], 
                        left_se=seL, right_se=seR, seP=None, s_in=s_in,
                        s_out=None, eta=eta_device, chemiPot=device.mu)
    green={}
    for t in range(len(tags)):
        green[tags[t]] = ans[t]

    assert list(green.keys())==['g_trans', 'grd', 'grl', 'gru', 'gr_left', 'gnd', 'gnl',\
                                            'gnu', 'gin_left', 'gpd', 'gpl', 'gpu', 'gip_left']


    assert list(green.keys())==['g_trans', 'grd', 'grl', 'gru', 'gr_left', 'gnd', 'gnl',\
                                    'gnu', 'gin_left', 'gpd', 'gpl', 'gpu', 'gip_left']
    g_trans= torch.tensor([[ 1.0983e-11-8.2022e-01j, -8.2022e-01+4.4634e-07j,8.9264e-07+8.2022e-01j,  8.2022e-01-1.3390e-06j],
        [-8.2022e-01+4.4634e-07j, -3.6607e-12-8.2022e-01j,-8.2021e-01+4.4631e-07j,  8.9264e-07+8.2022e-01j],
        [ 8.9264e-07+8.2022e-01j, -8.2021e-01+4.4631e-07j,-3.6607e-12-8.2022e-01j, -8.2022e-01+4.4634e-07j],
        [ 8.2022e-01-1.3390e-06j,  8.9264e-07+8.2022e-01j,-8.2022e-01+4.4634e-07j,  1.0983e-11-8.2022e-01j]],dtype=torch.complex128)
    grd= [torch.tensor([[ 1.0983e-11-8.2022e-01j, -8.2022e-01+4.4634e-07j,8.9264e-07+8.2022e-01j,  8.2022e-01-1.3390e-06j],
        [-8.2022e-01+4.4634e-07j, -3.6607e-12-8.2022e-01j,-8.2021e-01+4.4631e-07j,  8.9264e-07+8.2022e-01j],
        [ 8.9264e-07+8.2022e-01j, -8.2021e-01+4.4631e-07j,-3.6607e-12-8.2022e-01j, -8.2022e-01+4.4634e-07j],
        [ 8.2022e-01-1.3390e-06j,  8.9264e-07+8.2022e-01j,-8.2022e-01+4.4634e-07j,  1.0983e-11-8.2022e-01j]],dtype=torch.complex128)]

    assert  abs(g_trans-green['g_trans']).max()<1e-5
    assert  abs(grd[0]-green['grd'][0]).max()<1e-5
    assert green['grl'] == []
    assert green['gru'] == []

    gr_left= [torch.tensor([[ 1.0983e-11-8.2022e-01j, -8.2022e-01+4.4634e-07j,8.9264e-07+8.2022e-01j,  8.2022e-01-1.3390e-06j],
        [-8.2022e-01+4.4634e-07j, -3.6607e-12-8.2022e-01j,-8.2021e-01+4.4631e-07j,  8.9264e-07+8.2022e-01j],
        [ 8.9264e-07+8.2022e-01j, -8.2021e-01+4.4631e-07j,-3.6607e-12-8.2022e-01j, -8.2022e-01+4.4634e-07j],
        [ 8.2022e-01-1.3390e-06j,  8.9264e-07+8.2022e-01j,-8.2022e-01+4.4634e-07j,  1.0983e-11-8.2022e-01j]],dtype=torch.complex128)]

    gnd = [torch.tensor([[-2.2316e-07-4.1011e-01j,  1.2157e-13+2.2317e-07j,2.2316e-07+4.1011e-01j, -3.6432e-13-6.6949e-07j],
        [ 1.2132e-13+2.2317e-07j, -2.2316e-07-4.1011e-01j, 1.2154e-13+2.2315e-07j,  2.2316e-07+4.1011e-01j],
        [ 2.2316e-07+4.1011e-01j,  1.2132e-13+2.2315e-07j,-2.2316e-07-4.1011e-01j,  1.2149e-13+2.2317e-07j],
        [-3.6429e-13-6.6949e-07j,  2.2316e-07+4.1011e-01j,1.2140e-13+2.2317e-07j, -2.2316e-07-4.1011e-01j]],dtype=torch.complex128)]

    assert  abs(gr_left[0]-green['gr_left'][0]).max()<1e-5
    assert  abs(gnd[0]-green['gnd'][0]).max()<1e-5
    assert green['gnl'] == []
    assert green['gnu'] == []

    gin_left=[torch.tensor([[-2.2316e-07-4.1011e-01j,  1.2157e-13+2.2317e-07j,2.2316e-07+4.1011e-01j, -3.6432e-13-6.6949e-07j],
            [ 1.2132e-13+2.2317e-07j, -2.2316e-07-4.1011e-01j,1.2154e-13+2.2315e-07j,  2.2316e-07+4.1011e-01j],
            [ 2.2316e-07+4.1011e-01j,  1.2132e-13+2.2315e-07j,-2.2316e-07-4.1011e-01j,  1.2149e-13+2.2317e-07j],
            [-3.6429e-13-6.6949e-07j,  2.2316e-07+4.1011e-01j,1.2140e-13+2.2317e-07j, -2.2316e-07-4.1011e-01j]],dtype=torch.complex128)]
    assert  abs(gin_left[0]-green['gin_left'][0]).max()<1e-5

    assert green['gpd']== None
    assert green['gpl']== None
    assert green['gpu']== None
    assert green['gip_left']== None


