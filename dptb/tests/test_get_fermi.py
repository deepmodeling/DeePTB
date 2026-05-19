from dptb.postprocess.elec_struc_cal import ElecStruCal

# from dptb.postprocess.bandstructure.band import Band
from dptb.nn.build import build_model
import os
from pathlib import Path
import numpy as np

rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")


def test_get_fermi():
    ckpt = f"{rootdir}/test_get_fermi/nnsk.best.pth"  #  'hopping': {'method': 'poly2exp', 'rs': 5.0, 'w': 0.6},
    stru_data = f"{rootdir}/test_get_fermi/PRIMCELL.vasp"

    model = build_model(checkpoint=ckpt)
    nel_atom = {"Au":11}

    elec_cal = ElecStruCal(model=model,device='cpu')
    _, efermi =elec_cal.get_fermi_level(data=stru_data, 
                    nel_atom = nel_atom,smearing_method='FD',
                    meshgrid=[30,30,30],eig_solver='torch')
    assert abs(efermi  + 3.2262574434280395) < 1e-3

    _, efermi =elec_cal.get_fermi_level(data=stru_data, 
                    nel_atom = nel_atom,smearing_method='Gaussian',
                    meshgrid=[30,30,30],eig_solver='torch')
    assert abs(efermi  + 3.2262574434280395) < 1e-3

    _, efermi =elec_cal.get_fermi_level(data=stru_data, 
                    nel_atom = nel_atom,smearing_method='FD',
                    meshgrid=[30,30,30],eig_solver='numpy')
    assert abs(efermi  + 3.2262574434280395) < 1e-3

    _, efermi =elec_cal.get_fermi_level(data=stru_data, 
                    nel_atom = nel_atom,smearing_method='Gaussian',
                    meshgrid=[30,30,30],eig_solver='numpy')
    assert abs(efermi  + 3.2262574434280395) < 1e-3


def test_cal_E_fermi_ignores_invalid_padding():
    eigenvalues = np.array([[0.0, 1.0], [0.0, 1.0]])
    padded_eigenvalues = np.array([[0.0, 1.0, 1e4], [0.0, 1.0, 1e4]])
    valid_mask = np.array([[True, True, False], [True, True, False]])

    reference = ElecStruCal.cal_E_fermi(
        eigenvalues=eigenvalues,
        total_electrons=2,
        spindeg=2,
        wk=np.array([0.5, 0.5]),
    )
    masked = ElecStruCal.cal_E_fermi(
        eigenvalues=padded_eigenvalues,
        total_electrons=2,
        spindeg=2,
        wk=np.array([0.5, 0.5]),
        eigenvalue_valid_mask=valid_mask,
    )

    assert abs(reference - masked) < 1e-10
