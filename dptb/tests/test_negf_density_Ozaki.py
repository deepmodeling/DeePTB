# test_negf_density_Ozaki
from dptb.negf.density import Ozaki
from dptb.negf.device_property import DeviceProperty

from dptb.nn.build import build_model
import json
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
                                    torch_device = torch.device('cpu'),
                                    block_tridiagonal=negf_json['task_options']['block_tridiagonal'])

    # hamiltonian = NEGFHamiltonianInit(apiH=apiHrk, structase=structase, stru_options=task_options["stru_options"], results_path=results_path)
    kpoints= kmesh_sampling(negf_json['task_options']["stru_options"]["kmesh"])
    with torch.no_grad():
        struct_device, struct_leads, _,_,_ = hamiltonian.initialize(kpoints=kpoints)

    
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


    # check Ozaki
    kpoints= kmesh_sampling(negf_json['task_options']["stru_options"]["kmesh"])
    density_options = j_must_have(negf_json['task_options'], "density_options")

    density = Ozaki(R=density_options["R"], M_cut=density_options["M_cut"], n_gauss=density_options["n_gauss"])

    #compute_density
    DM_eq, DM_neq = density.integrate(deviceprop=deviceprop, kpoint=kpoints[0])
    DM_eq_standard = torch.tensor([[ 1.0000e+00, -6.3615e-01,  3.4565e-07,  2.1080e-01],
            [-6.3615e-01,  1.0000e+00, -6.3615e-01,  3.4565e-07],
            [ 3.4565e-07, -6.3615e-01,  1.0000e+00, -6.3615e-01],
            [ 2.1080e-01,  3.4565e-07, -6.3615e-01,  1.0000e+00]],dtype=torch.float64)
    
    assert np.array(abs(DM_eq_standard-DM_eq)<1e-5).all()       
    assert DM_neq==0.0     

    onsite_density=density.get_density_onsite(deviceprop=deviceprop,DM=DM_eq)
    onsite_density_standard = torch.tensor([[ 0.0000,  0.0000,  6.4000,  1.0000],[ 0.0000,  0.0000,  8.0000,  1.0000],
        [ 0.0000,  0.0000,  9.6000,  1.0000],[ 0.0000,  0.0000, 11.2000,  1.0000]], dtype=torch.float64)
    assert np.array(abs(onsite_density_standard-onsite_density)<1e-5).all()