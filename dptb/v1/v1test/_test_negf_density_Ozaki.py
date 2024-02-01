# test_negf_density_Ozaki
from dptb.negf.density import Ozaki
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
import pytest



@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    """
    :return:
    """
    return str(request.config.rootdir)

def test_negf_density_Ozaki(root_directory):

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
            "results_path":root_directory +"/dptb/tests/data/test_negf",
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


    # check Ozaki
    kpoints= kmesh_sampling(task_options["stru_options"]["kmesh"])
    density_options = j_must_have(task_options, "density_options")

    density = Ozaki(R=density_options["R"], M_cut=density_options["M_cut"], n_gauss=density_options["n_gauss"])

    #compute_density
    DM_eq, DM_neq = density.integrate(deviceprop=device, kpoint=kpoints[0])
    DM_eq_standard = torch.tensor([[ 1.0000e+00, -6.3615e-01,  3.4565e-07,  2.1080e-01],
            [-6.3615e-01,  1.0000e+00, -6.3615e-01,  3.4565e-07],
            [ 3.4565e-07, -6.3615e-01,  1.0000e+00, -6.3615e-01],
            [ 2.1080e-01,  3.4565e-07, -6.3615e-01,  1.0000e+00]],dtype=torch.float64)

    assert np.array(abs(DM_eq_standard-DM_eq)<1e-5).all()       
    assert DM_neq==0.0     

    onsite_density=density.get_density_onsite(deviceprop=device,DM=DM_eq)
    onsite_density_standard = torch.tensor([[ 0.0000,  0.0000,  6.4000,  1.0000],[ 0.0000,  0.0000,  8.0000,  1.0000],
        [ 0.0000,  0.0000,  9.6000,  1.0000],[ 0.0000,  0.0000, 11.2000,  1.0000]], dtype=torch.float64)
    assert np.array(abs(onsite_density_standard-onsite_density)<1e-5).all()