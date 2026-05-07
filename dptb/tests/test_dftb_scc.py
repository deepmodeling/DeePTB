import torch
import os
from pathlib import Path
from dptb.nn.dftb.dftb_scc import DFTBSCC, SKSCC
from dptb.nn.dftb.scc_params import SCCParams
from dptb.nn.dftb.sk_param import SKParam
from dptb.nn.dftbsk import DFTBSK
from dptb.nn.nnsk import NNSK
from dptb.data import AtomicDataDict
import pytest
import numpy as np


rootdir = Path(__file__).resolve().parent / "data"

SCALES = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15], dtype=float)
BENCHMARK_STRUCT_ROOT = rootdir / "dftb" / "structs_eos"
DFTBP_ROOT = BENCHMARK_STRUCT_ROOT / "dftbp_results"
SK_PATH = rootdir / "dftb"
DFTB_SCC_BENCHMARK_SETUP = {
    "basis": {"C": ["2s", "2p"]},
    "nel_atom": {"C": 4},
    "sigma_rep": {"C": 0.561},
    "AtomicData_options": {"r_max": {"C": 4.2}},
    "tol": 1e-14,
    "mix_rate": 0.30,
    "max_iter": 1000,
    "Temp": 0.1,
    "smearing_method": "FD",
    "mixer": "pulay",
    "overlap": True,
    "smooth_ski": True,
}
KPOINTS_BY_STRUCT = {
    "BCC": [20, 20, 20],
    "GRAPHENE": [50, 50, 1],
    "DIMER": [1, 1, 1],
}
STRUCT_FILE_BY_STRUCT = {
    "BCC": "POSCAR",
    "GRAPHENE": "POSCAR",
    "DIMER": "POSCAR",
}
ELECTRONIC_ENERGY_ATOL = 1e-2  # require the electronic energies to match within 10 meV


def test_dftbscc_explicit_dftbsk_and_scc_params(rootdir=rootdir):
    sk_path = os.path.join(rootdir, 'dftb')
    basis = {'B': ['2s', '2p'], 'N': ['2s', '2p']}
    model = DFTBSK(basis=basis, skdata=sk_path, overlap=True, dtype=torch.float64)
    skp = SKParam(basis=basis, skdata=sk_path, cal_rcuts=True, dtype=torch.float64)
    scc_params = SCCParams.from_skparam(skp)

    scc = SKSCC(model=model, params=scc_params, overlap=True)

    assert scc.model is model
    assert scc.scc_params is scc_params
    assert scc.r_max == scc_params.r_max


def test_dftbscc_nnsk_scc_options_init():
    basis = {"H": ["1s"]}
    model = NNSK(basis=basis, overlap=True, hopping={"method": "powerlaw", "rs": 3.0, "w": 0.2})

    params = SCCParams.from_options(
        basis=basis,
        idp_sk=model.idp_sk,
        options={
            "hubbard_u": {"H": {"1s": 10.0}},
            "occupation": {"H": {"1s": 1}},
            "highest_occu_u": {"H": 10.0},
            "mass": {"H": 1.008},
            "use_database": False,
        },
        model=model,
    )
    scc = SKSCC(model=model, params=params)

    assert scc.scc_params.skdict["HubdU"][0, 0, 0] == 10.0
    assert scc.r_max == {"H": 3.0}


def test_dftbscc_nnsk_database_and_overlap_checks():
    basis = {"H": ["1s"]}
    model = NNSK(basis=basis, overlap=True, hopping={"method": "powerlaw", "rs": 3.0, "w": 0.2})
    params = SCCParams.from_options(basis=basis, idp_sk=model.idp_sk, options={"use_database": True}, model=model)
    scc = SKSCC(model=model, params=params)
    assert scc.scc_params.skdict["HubdU"][0, 0, 0] > 0

    model_without_overlap = NNSK(basis=basis, overlap=False, hopping={"method": "powerlaw", "rs": 3.0, "w": 0.2})
    orthogonal_scc = SKSCC(model=model_without_overlap, params=params)
    assert orthogonal_scc.overlap is False
    assert orthogonal_scc.mulliken.overlap is False

    with pytest.raises(ValueError, match="overlap=True"):
        SKSCC(model=model_without_overlap, params=params, overlap=True)


def test_skscc_can_force_orthogonal_mode_for_overlap_model():
    basis = {"H": ["1s"]}
    model = NNSK(basis=basis, overlap=True, hopping={"method": "powerlaw", "rs": 3.0, "w": 0.2})
    params = SCCParams.from_options(basis=basis, idp_sk=model.idp_sk, options={"use_database": True}, model=model)

    scc = SKSCC(model=model, params=params, overlap=False)

    assert scc.overlap is False
    assert scc.mulliken.overlap is False


def test_skscc_cal_scc_hk_orthogonal_has_diagonal_correction_only():
    basis = {"H": ["1s"]}
    model = NNSK(basis=basis, overlap=False, hopping={"method": "powerlaw", "rs": 3.0, "w": 0.2})
    params = SCCParams.from_options(basis=basis, idp_sk=model.idp_sk, options={"use_database": True}, model=model)
    scc = SKSCC(model=model, params=params)
    data = {
        AtomicDataDict.HAMILTONIAN_KEY: torch.zeros((1, 3, 3), dtype=torch.float64),
    }
    per_atom_indices = np.array([0, 1, 3])
    scc_shift = torch.tensor([2.0, 4.0], dtype=torch.float64)

    scc_hk = scc.cal_scc_hk(data=data, per_atom_indices=per_atom_indices, scc_shift=scc_shift)

    expected = torch.diag(torch.tensor([2.0, 4.0, 4.0], dtype=torch.complex128)).unsqueeze(0)
    assert torch.allclose(scc_hk, expected)


def test_skscc_and_dftbscc_wrapper_hbn_equivalent(rootdir=rootdir):
    sk_path = os.path.join(rootdir, 'dftb')
    struct = os.path.join(rootdir, 'hBN/hBN.vasp')
    basis = {'B': ['2s', '2p'], 'N': ['2s', '2p']}
    atomic_data_options = {"r_max": {'B': 6.349479778742587, 'N': 5.366822193937187}}
    run_options = {
        "data": struct,
        "nel_atom": {'B': 3, 'N': 5},
        "kmeshgrid": [20, 20, 1],
        "kgamma_center": True,
        "krotational_symmetry": True,
        "ktime_inversion_symmetry": True,
        "AtomicData_options": atomic_data_options,
        "mix_rate": 0.25,
        "max_iter": 1000,
        "smearing_method": 'Fermi-Dirac',
    }

    model = DFTBSK(basis=basis, skdata=sk_path, overlap=True, dtype=torch.float64)
    skp = SKParam(basis=basis, skdata=sk_path, cal_rcuts=True, dtype=torch.float64)
    scc = SKSCC(model=model, params=SCCParams.from_skparam(skp), overlap=True)
    wrapper = DFTBSCC(basis=basis, sk_path=sk_path, overlap=True)

    scc.run_iters(**run_options)
    wrapper.run_iters(**run_options)

    assert torch.allclose(scc.elec_totE, wrapper.elec_totE, atol=1e-10)
    assert np.allclose(scc.mulliken.mul_charge, wrapper.mulliken.mul_charge, atol=1e-10)

def test_skscc_get_total_energy_requires_repulsive_params(rootdir=rootdir):
    sk_path = os.path.join(rootdir, 'dftb')
    basis = {'B': ['2s', '2p'], 'N': ['2s', '2p']}
    model = DFTBSK(basis=basis, skdata=sk_path, overlap=True, dtype=torch.float64)
    skp = SKParam(basis=basis, skdata=sk_path, cal_rcuts=True, dtype=torch.float64)
    scc = SKSCC(model=model, params=SCCParams.from_skparam(skp), overlap=True)

    with pytest.raises(ValueError, match="Repulsive parameters are required"):
        scc.get_total_energy(
            data=os.path.join(rootdir, 'hBN/hBN.vasp'),
            nel_atom={'B': 3, 'N': 5},
            kmeshgrid=[1, 1, 1],
            AtomicData_options={"r_max": {'B': 6.349479778742587, 'N': 5.366822193937187}},
            max_iter=1,
        )



def test_dftbscc_hBN(rootdir = rootdir):


    sk_path = os.path.join(rootdir, 'dftb')
    struct = os.path.join(rootdir, 'hBN/hBN.vasp')
    basis = {'B':['2s','2p'],"N":["2s","2p"]}
    AtomicData_options = {"r_max": {'B': 6.349479778742587, 'N': 5.366822193937187}}
    overlap = True
    kmeshgrid = [20, 20, 1]  # Example k-point mesh grid
    nel_atom = {'B': 3, 'N': 5}


    dftbscc = DFTBSCC(  basis = basis,
                        sk_path = sk_path,
                        overlap = overlap)

    dftbscc.run_iters(data = struct,
                    nel_atom = nel_atom,
                    kmeshgrid = kmeshgrid,
                    kgamma_center = True,
                    krotational_symmetry = True,
                    ktime_inversion_symmetry = True,
                    AtomicData_options = AtomicData_options,
                    mix_rate = 0.25,
                    max_iter = 1000,
                    smearing_method = 'Fermi-Dirac')

    per_atom_charge_ref = [5,3]
    delta_charge_ref = np.array([ 0.22519684, -0.22519661])
    mulliken_ref = np.array([5.22519684, 2.77480339])
    scc_shift_ref = torch.tensor([0.6616, 0.2669], dtype=torch.float64)
    Gamma_ref = torch.tensor([[71.8542, 68.9163],
                                [68.9163, 67.7311]], dtype=torch.float64)
    expGamma_ref = torch.tensor([1.2838e-04, 4.5020e-04, 7.5969e-04, 4.5020e-04, 1.2838e-04, 9.7802e-03,
        4.5020e-04, 4.5020e-04, 9.7802e-03, 1.2562e-04, 7.5969e-04, 9.7801e-03,
        8.7077e-02, 7.5477e-02, 4.5020e-04, 8.7077e-02, 9.7801e-03, 7.5969e-04,
        1.2562e-04, 1.7459e+00, 7.5478e-02, 4.5020e-04, 7.5477e-02, 1.2838e-04,
        9.7801e-03, 8.7077e-02, 1.7459e+00, 1.7459e+00, 9.7802e-03, 1.2562e-04,
        2.7461e-01, 1.8722e-03, 7.2770e-03, 2.7461e-01, 2.7461e-01, 7.2770e-03,
        1.8722e-03, 7.2770e-03, 1.8722e-03, 1.2838e-04, 4.5020e-04, 7.5969e-04,
        4.5020e-04, 1.2838e-04, 9.7802e-03, 4.5020e-04, 4.5020e-04, 9.7802e-03,
        1.2562e-04, 7.5969e-04, 9.7801e-03, 8.7077e-02, 7.5477e-02, 4.5020e-04,
        8.7077e-02, 9.7801e-03, 7.5969e-04, 1.2562e-04, 1.7459e+00, 7.5478e-02,
        4.5020e-04, 7.5477e-02, 1.2838e-04, 9.7801e-03, 8.7077e-02, 1.7459e+00,
        1.7459e+00, 9.7802e-03, 1.2562e-04, 2.7461e-01, 1.8722e-03, 7.2770e-03,
        2.7461e-01, 2.7461e-01, 7.2770e-03, 1.8722e-03, 7.2770e-03, 1.8722e-03], dtype=torch.float64)
    expGamma_ref_sorted, _ = torch.sort(expGamma_ref)
     # in different version, the order of expGamma may change, so we sort it before comparison
    expGamma_sorted, _ = torch.sort(dftbscc.expGamma)
    expGamma_onsite_ref = torch.tensor([13.3009, 10.3525], dtype=torch.float64)
    inv_r_ref = torch.tensor([[59.0811, 74.4423],
        [74.4423, 59.0811]], dtype=torch.float64)

    assert np.allclose(dftbscc.mulliken.mul_charge, mulliken_ref, atol=1e-5)
    assert torch.allclose(dftbscc.scc_shift, scc_shift_ref, atol=1e-4)
    assert torch.allclose(dftbscc.Gamma, Gamma_ref, atol=1e-4)
    assert torch.allclose(dftbscc.expGamma_onsite, expGamma_onsite_ref, atol=1e-4)
    assert torch.allclose(dftbscc.inv_r, inv_r_ref, atol=1e-4)
    assert torch.allclose(expGamma_sorted, expGamma_ref_sorted, atol=1e-4)
    assert np.allclose(dftbscc.mulliken.per_atom_charge, per_atom_charge_ref, atol=1e-5)
    assert np.allclose(dftbscc.mulliken.delta_charge, delta_charge_ref, atol=1e-5)


    # Test energy terms
    # Reference values updated after fixing k-point weight handling in get_fermi_level
    elec_H0_bandE_ref = torch.tensor([-104.61467950], dtype=torch.float64)
    scc_shift_energy_ref = torch.tensor(0.04444195, dtype=torch.float64)
    elec_totE_ref = torch.tensor([-104.57023755], dtype=torch.float64)

    # Expected values from DFTB+ (for reference)
    # Energy H0: -104.5893
    # SCC shift energy: 0.0466
    # Total energy: -104.5427

    assert dftbscc.elec_H0_bandE is not None, "elec_H0_bandE should not be None after convergence"
    assert dftbscc.scc_shift_energy is not None, "scc_shift_energy should not be None after convergence"
    assert dftbscc.elec_totE is not None, "elec_totE should not be None after convergence"
    assert torch.allclose(dftbscc.elec_H0_bandE, elec_H0_bandE_ref, atol=1e-6)
    assert torch.allclose(dftbscc.scc_shift_energy, scc_shift_energy_ref, atol=1e-6)
    assert torch.allclose(dftbscc.elec_totE, elec_totE_ref, atol=1e-6)
    # Verify the relationship: elec_totE = elec_H0_bandE + scc_shift_energy
    assert torch.allclose(dftbscc.elec_totE, dftbscc.elec_H0_bandE + dftbscc.scc_shift_energy, atol=1e-10)


def _benchmark_struct_paths(struct_type: str):
    base = BENCHMARK_STRUCT_ROOT / f"C_{struct_type}"
    prefix = f"C_{struct_type.lower()}_"
    file_name = STRUCT_FILE_BY_STRUCT[struct_type]
    return [base / f"{prefix}{scale:.3f}" / file_name for scale in SCALES]


def _load_dftbp_electronic(struct_type: str):
    if struct_type == "BCC":
        path = os.path.join(DFTBP_ROOT, "eos_data_bcc.txt")
    elif struct_type == "GRAPHENE":
        path = os.path.join(DFTBP_ROOT, "eos_data_gra.txt")
    elif struct_type == "DIMER":
        path = os.path.join(DFTBP_ROOT, "potential_energy_data_dimer.txt")
    else:
        raise ValueError(f"Unsupported structure type {struct_type}")
    data = np.genfromtxt(path, comments="#")
    assert np.allclose(data[:, 0], SCALES, atol=1e-6), "Scale grid mismatch against reference"
    return data[:, 3]


def _run_deeptb_scc_electronic(struct_type: str):
    struct_paths = _benchmark_struct_paths(struct_type)
    results = []
    for path in struct_paths:
        dftbscc = DFTBSCC(
            basis=DFTB_SCC_BENCHMARK_SETUP["basis"],
            sk_path=str(SK_PATH),
            overlap=DFTB_SCC_BENCHMARK_SETUP["overlap"],
            smooth_ski=DFTB_SCC_BENCHMARK_SETUP["smooth_ski"],
        )
        dftbscc.get_total_energy(
            data=str(path),
            nel_atom=DFTB_SCC_BENCHMARK_SETUP["nel_atom"],
            sigma_rep=DFTB_SCC_BENCHMARK_SETUP["sigma_rep"],
            kmeshgrid=KPOINTS_BY_STRUCT[struct_type],
            kgamma_center=True,
            krotational_symmetry=False,
            ktime_inversion_symmetry=True,
            tol=DFTB_SCC_BENCHMARK_SETUP["tol"],
            mix_rate=DFTB_SCC_BENCHMARK_SETUP["mix_rate"],
            max_iter=DFTB_SCC_BENCHMARK_SETUP["max_iter"],
            Temp=DFTB_SCC_BENCHMARK_SETUP["Temp"],
            AtomicData_options=DFTB_SCC_BENCHMARK_SETUP["AtomicData_options"],
            smearing_method=DFTB_SCC_BENCHMARK_SETUP["smearing_method"],
            mixer=DFTB_SCC_BENCHMARK_SETUP["mixer"],
        )
        results.append(float(dftbscc.elec_totE))
    return np.array(results, dtype=float)


@pytest.mark.parametrize("struct_type", ["BCC", "GRAPHENE", "DIMER"])
def test_dftbscc_matches_dftbp_benchmarks(struct_type):
    """Benchmark DeePTB SCC electronic energies against DFTB+ references."""
    dftbp_elec = _load_dftbp_electronic(struct_type)
    deeptb_elec = _run_deeptb_scc_electronic(struct_type)
    assert deeptb_elec.shape == dftbp_elec.shape
    per_scale_diff = deeptb_elec - dftbp_elec
    max_diff = np.max(np.abs(per_scale_diff))
    assert max_diff < ELECTRONIC_ENERGY_ATOL, (
        f"{struct_type} electronic energies deviate more than {ELECTRONIC_ENERGY_ATOL} eV; "
        f"max_abs_diff={max_diff}; per_scale_diff={list(zip(SCALES.tolist(), per_scale_diff.tolist()))}"
    )


def test_dftbscc_CH4(rootdir = rootdir):


    sk_path = os.path.join(rootdir, 'dftb')
    struct = os.path.join(rootdir, 'ch4/CH4.vasp')
    basis = {'C':['2s','2p'],"H":["1s"]}
    AtomicData_options = {"r_max": {'C': 6.2, 'H': 4.5}}
    overlap = True
    kmeshgrid = [1, 1, 1]  # Example k-point mesh grid
    nel_atom = {'C': 4, 'H': 1}


    dftbscc = DFTBSCC(  basis = basis,
                    sk_path = sk_path,
                    overlap = True)

    dftbscc.run_iters(data = struct,
                    nel_atom = nel_atom,
                    kmeshgrid = kmeshgrid,
                    kgamma_center = True,
                    krotational_symmetry = True,
                    ktime_inversion_symmetry = True,
                    AtomicData_options = AtomicData_options,
                    mix_rate = 0.25,
                    max_iter = 1000,
                    smearing_method = 'Fermi-Dirac')
    
    per_atom_charge_ref = [4,1,1,1,1]
    delta_charge_ref = np.array([ 0.37197185, -0.07783353, -0.10351726, -0.09531036, -0.09531026])
    mulliken_ref = np.array([4.37194607, 0.92217521, 0.89649291, 0.90469262, 0.90469262])
    scc_shift_ref = torch.tensor([0.4431, 0.2590, 0.2212, 0.2293, 0.2293], dtype=torch.float64)
    Gamma_ref = torch.tensor([[8.4996, 6.9566, 7.4890, 7.3541, 7.3541],
        [6.9566, 9.3724, 5.4857, 5.4101, 5.4101],
        [7.4890, 5.4857, 9.3724, 6.1239, 6.1239],
        [7.3541, 5.4101, 6.1239, 9.3724, 5.8532],
        [7.3541, 5.4101, 6.1239, 5.8532, 9.3724]], dtype=torch.float64)
    expGamma_ref = torch.tensor([4.2158, 6.5593, 5.8384, 5.8384, 1.1051, 1.0413, 1.0413, 1.7955, 1.7955,
        1.4659, 4.2158, 6.5593, 5.8384, 5.8384, 1.1051, 1.0413, 1.0413, 1.7955,
        1.7955, 1.4659], dtype=torch.float64)
    expGamma_onsite_ref = torch.tensor([10.5423, 11.4152, 11.4152, 11.4152, 11.4152], dtype=torch.float64)
    inv_r_ref = torch.tensor([[-2.0428, 11.1724, 14.0483, 13.1925, 13.1925],
        [11.1724, -2.0428,  6.5908,  6.4515,  6.4515],
        [14.0483,  6.5908, -2.0428,  7.9194,  7.9194],
        [13.1925,  6.4515,  7.9194, -2.0428,  7.3191],
        [13.1925,  6.4515,  7.9194,  7.3191, -2.0428]], dtype=torch.float64)
        
    assert np.allclose(dftbscc.mulliken.per_atom_charge, per_atom_charge_ref, atol=1e-5)
    # Tolerances slightly relaxed to accommodate double precision changes
    assert np.allclose(dftbscc.mulliken.delta_charge, delta_charge_ref, atol=5e-5)
    assert np.allclose(dftbscc.mulliken.mul_charge, mulliken_ref, atol=5e-5)
    assert torch.allclose(dftbscc.scc_shift, scc_shift_ref, atol=2e-4)
    assert torch.allclose(dftbscc.Gamma, Gamma_ref, atol=3e-4)
    expGamma_sorted, _ = torch.sort(dftbscc.expGamma)
    expGamma_ref_sorted, _ = torch.sort(expGamma_ref)
    assert torch.allclose(expGamma_sorted, expGamma_ref_sorted, atol=3e-4)
    assert torch.allclose(dftbscc.expGamma_onsite, expGamma_onsite_ref, atol=1e-4)
    assert torch.allclose(dftbscc.inv_r, inv_r_ref, atol=3e-4)

    # Test energy terms
    elec_H0_bandE_ref = torch.tensor([-90.823863], dtype=torch.float64)
    scc_shift_energy_ref = torch.tensor(0.039016, dtype=torch.float64)
    elec_totE_ref = torch.tensor([-90.784848], dtype=torch.float64)

    assert dftbscc.elec_H0_bandE is not None, "elec_H0_bandE should not be None after convergence"
    assert dftbscc.scc_shift_energy is not None, "scc_shift_energy should not be None after convergence"
    assert dftbscc.elec_totE is not None, "elec_totE should not be None after convergence"
    # Tolerances slightly relaxed to accommodate double precision changes
    assert torch.allclose(dftbscc.elec_H0_bandE, elec_H0_bandE_ref, atol=1e-5)
    assert torch.allclose(dftbscc.scc_shift_energy, scc_shift_energy_ref, atol=2e-5)
    assert torch.allclose(dftbscc.elec_totE, elec_totE_ref, atol=1e-5)
    # Verify the relationship: elec_totE = elec_H0_bandE + scc_shift_energy
    assert torch.allclose(dftbscc.elec_totE, dftbscc.elec_H0_bandE + dftbscc.scc_shift_energy, atol=1e-10)


def test_eigh_solver_no_overlap():
    # Create a Hermitian matrix
    H = torch.tensor([[[2.0, 1.0], [1.0, 3.0]]], dtype=torch.float64)
    eigvecs, eigvals = DFTBSCC.eigh_solver(H_mat=H)
    # Check shapes
    assert len(eigvecs) == 1
    assert len(eigvals) == 1
    assert eigvecs[0].shape == (1, 2, 2)
    assert eigvals[0].shape == (1, 2)
    # Check eigenvalues
    expected_eigvals = torch.linalg.eigh(H)[0]
    assert torch.allclose(eigvals[0], expected_eigvals)

def test_eigh_solver_with_overlap():
    # Hermitian H and positive-definite overlap matrix S
    H = torch.tensor([[[2.0, 1.0], [1.0, 3.0]]], dtype=torch.float64)
    S = torch.tensor([[[2.0, 0.5], [0.5, 1.5]]], dtype=torch.float64)
    eigvecs, eigvals = DFTBSCC.eigh_solver(H_mat=H, overlap=True, overlap_mat=S)
    # Check shapes
    assert len(eigvecs) == 1
    assert len(eigvals) == 1
    assert eigvecs[0].shape == (1, 2, 2)
    assert eigvals[0].shape == (1, 2)

def test_eigh_solver_overlap_missing_matrix():
    H = torch.tensor([[[2.0, 1.0], [1.0, 3.0]]], dtype=torch.float64)
    with pytest.raises(ValueError):
        DFTBSCC.eigh_solver(H_mat=H, overlap=True)

def test_eigh_solver_rejects_unused_overlap_matrix():
    H = torch.tensor([[[2.0, 1.0], [1.0, 3.0]]], dtype=torch.float64)
    S = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    with pytest.raises(ValueError, match="overlap_mat"):
        DFTBSCC.eigh_solver(H_mat=H, overlap=False, overlap_mat=S)

def test_eigh_solver_complex_matrix():
    # Hermitian complex matrix
    H = torch.zeros((1, 2, 2), dtype=torch.complex64)
    H[0, 0, 0] = 2 + 0j
    H[0, 0, 1] = 1 - 1j
    H[0, 1, 0] = 1 + 1j
    H[0, 1, 1] = 3 + 0j
    eigvecs, eigvals = DFTBSCC.eigh_solver(H_mat=H)
    assert eigvecs[0].shape == (1, 2, 2)
    assert eigvals[0].shape == (1, 2)


def test_dftbscc_reset_clears_state(rootdir=rootdir):
    """Test that reset() properly clears all per-calculation state variables."""
    sk_path = os.path.join(rootdir, 'dftb')
    struct = os.path.join(rootdir, 'hBN/hBN.vasp')
    basis = {'B': ['2s', '2p'], "N": ["2s", "2p"]}
    AtomicData_options = {"r_max": {'B': 6.349479778742587, 'N': 5.366822193937187}}
    kmeshgrid = [20, 20, 1]
    nel_atom = {'B': 3, 'N': 5}

    dftbscc = DFTBSCC(basis=basis, sk_path=sk_path, overlap=True)

    # Run a calculation to populate state
    dftbscc.run_iters(
        data=struct,
        nel_atom=nel_atom,
        kmeshgrid=kmeshgrid,
        kgamma_center=True,
        krotational_symmetry=False,
        ktime_inversion_symmetry=True,
        AtomicData_options=AtomicData_options,
        max_iter=1000
    )

    # Verify state is populated after calculation
    assert dftbscc.atomic_numbers is not None, "atomic_numbers should be set after calculation"
    assert dftbscc.elec_totE is not None, "elec_totE should be set after calculation"
    assert dftbscc.elec_H0_bandE is not None, "elec_H0_bandE should be set after calculation"
    assert dftbscc.E_fermi is not None, "E_fermi should be set after calculation"
    assert dftbscc.scc_shift is not None, "scc_shift should be set after calculation"
    assert dftbscc.scc_shift_energy is not None, "scc_shift_energy should be set after calculation"
    assert dftbscc.data is not None, "data should be set after calculation"
    assert dftbscc.expGamma is not None, "expGamma should be set after calculation"
    assert dftbscc.expGamma_onsite is not None, "expGamma_onsite should be set after calculation"
    assert dftbscc.inv_r is not None, "inv_r should be set after calculation"
    assert dftbscc.Gamma is not None, "Gamma should be set after calculation"

    # Call reset explicitly
    dftbscc.reset()

    # Verify all per-calculation state variables are None after reset
    assert dftbscc.atomic_numbers is None, "atomic_numbers should be None after reset"
    assert dftbscc.elec_totE is None, "elec_totE should be None after reset"
    assert dftbscc.elec_H0_bandE is None, "elec_H0_bandE should be None after reset"
    assert dftbscc.elec_bandE is None, "elec_bandE should be None after reset"
    assert dftbscc.E_fermi is None, "E_fermi should be None after reset"
    assert dftbscc.mulcharge_old is None, "mulcharge_old should be None after reset"
    assert dftbscc.scc_shift is None, "scc_shift should be None after reset"
    assert dftbscc.scc_shift_energy is None, "scc_shift_energy should be None after reset"
    assert dftbscc.data is None, "data should be None after reset"
    assert dftbscc.expGamma is None, "expGamma should be None after reset"
    assert dftbscc.expGamma_onsite is None, "expGamma_onsite should be None after reset"
    assert dftbscc.inv_r is None, "inv_r should be None after reset"
    assert dftbscc.Gamma is None, "Gamma should be None after reset"
    assert dftbscc.total_energy is None, "total_energy should be None after reset"
    assert dftbscc.total_rep_energy is None, "total_rep_energy should be None after reset"

    # Verify persistent state (model, skp, etc.) is NOT cleared
    assert dftbscc.model is not None, "model should persist after reset"
    assert dftbscc.skp is not None, "skp should persist after reset"
    assert dftbscc.mulliken is not None, "mulliken should persist after reset"
    assert dftbscc.h2k is not None, "h2k should persist after reset"
    assert dftbscc.s2k is not None, "s2k should persist after reset"


def test_dftbscc_reuse_gives_identical_results(rootdir=rootdir):
    """Test that reusing DFTBSCC instance gives identical results to fresh instances.

    This is the key test for validating that reset() works correctly:
    - Running the same structure twice with the same instance should give identical results
    - Results should also match a fresh instance
    """
    sk_path = os.path.join(rootdir, 'dftb')
    struct = os.path.join(rootdir, 'hBN/hBN.vasp')
    basis = {'B': ['2s', '2p'], "N": ["2s", "2p"]}
    AtomicData_options = {"r_max": {'B': 6.349479778742587, 'N': 5.366822193937187}}
    kmeshgrid = [20, 20, 1]
    nel_atom = {'B': 3, 'N': 5}
    run_params = {
        'nel_atom': nel_atom,
        'kmeshgrid': kmeshgrid,
        'kgamma_center': True,
        'krotational_symmetry': False,
        'ktime_inversion_symmetry': True,
        'AtomicData_options': AtomicData_options,
        'max_iter': 1000
    }

    # First run with instance 1
    dftbscc1 = DFTBSCC(basis=basis, sk_path=sk_path, overlap=True)
    dftbscc1.run_iters(data=struct, **run_params)

    # Store results from first run
    elec_totE_run1 = dftbscc1.elec_totE.clone()
    elec_H0_bandE_run1 = dftbscc1.elec_H0_bandE.clone()
    scc_shift_energy_run1 = dftbscc1.scc_shift_energy.clone()
    scc_shift_run1 = dftbscc1.scc_shift.clone()
    Gamma_run1 = dftbscc1.Gamma.clone()
    mul_charge_run1 = dftbscc1.mulliken.mul_charge.copy()

    # Second run with SAME instance (tests that internal reset works)
    dftbscc1.run_iters(data=struct, **run_params)

    # Store results from second run
    elec_totE_run2 = dftbscc1.elec_totE.clone()
    elec_H0_bandE_run2 = dftbscc1.elec_H0_bandE.clone()
    scc_shift_energy_run2 = dftbscc1.scc_shift_energy.clone()
    scc_shift_run2 = dftbscc1.scc_shift.clone()
    Gamma_run2 = dftbscc1.Gamma.clone()
    mul_charge_run2 = dftbscc1.mulliken.mul_charge.copy()

    # Third run with a FRESH instance
    dftbscc2 = DFTBSCC(basis=basis, sk_path=sk_path, overlap=True)
    dftbscc2.run_iters(data=struct, **run_params)

    # Store results from fresh instance
    elec_totE_run3 = dftbscc2.elec_totE.clone()
    elec_H0_bandE_run3 = dftbscc2.elec_H0_bandE.clone()
    scc_shift_energy_run3 = dftbscc2.scc_shift_energy.clone()
    scc_shift_run3 = dftbscc2.scc_shift.clone()
    Gamma_run3 = dftbscc2.Gamma.clone()
    mul_charge_run3 = dftbscc2.mulliken.mul_charge.copy()

    # Verify run 1 and run 2 (reused instance) give identical results
    assert torch.allclose(elec_totE_run1, elec_totE_run2, atol=1e-10), \
        f"Reused instance elec_totE differs: {elec_totE_run1} vs {elec_totE_run2}"
    assert torch.allclose(elec_H0_bandE_run1, elec_H0_bandE_run2, atol=1e-10), \
        f"Reused instance elec_H0_bandE differs: {elec_H0_bandE_run1} vs {elec_H0_bandE_run2}"
    assert torch.allclose(scc_shift_energy_run1, scc_shift_energy_run2, atol=1e-10), \
        f"Reused instance scc_shift_energy differs: {scc_shift_energy_run1} vs {scc_shift_energy_run2}"
    assert torch.allclose(scc_shift_run1, scc_shift_run2, atol=1e-10), \
        f"Reused instance scc_shift differs: {scc_shift_run1} vs {scc_shift_run2}"
    assert torch.allclose(Gamma_run1, Gamma_run2, atol=1e-10), \
        f"Reused instance Gamma differs"
    assert np.allclose(mul_charge_run1, mul_charge_run2, atol=1e-10), \
        f"Reused instance mul_charge differs: {mul_charge_run1} vs {mul_charge_run2}"

    # Verify run 1 and run 3 (fresh instance) give identical results
    assert torch.allclose(elec_totE_run1, elec_totE_run3, atol=1e-10), \
        f"Fresh instance elec_totE differs: {elec_totE_run1} vs {elec_totE_run3}"
    assert torch.allclose(elec_H0_bandE_run1, elec_H0_bandE_run3, atol=1e-10), \
        f"Fresh instance elec_H0_bandE differs: {elec_H0_bandE_run1} vs {elec_H0_bandE_run3}"
    assert torch.allclose(scc_shift_energy_run1, scc_shift_energy_run3, atol=1e-10), \
        f"Fresh instance scc_shift_energy differs: {scc_shift_energy_run1} vs {scc_shift_energy_run3}"
    assert torch.allclose(scc_shift_run1, scc_shift_run3, atol=1e-10), \
        f"Fresh instance scc_shift differs: {scc_shift_run1} vs {scc_shift_run3}"
    assert torch.allclose(Gamma_run1, Gamma_run3, atol=1e-10), \
        f"Fresh instance Gamma differs"
    assert np.allclose(mul_charge_run1, mul_charge_run3, atol=1e-10), \
        f"Fresh instance mul_charge differs: {mul_charge_run1} vs {mul_charge_run3}"


# ============================================================================
# Tests for scc_dtype parameter
# ============================================================================

def test_scc_dtype_default_is_float64(rootdir=rootdir):
    """Test that the default scc_dtype is torch.float64."""
    sk_path = os.path.join(rootdir, 'dftb')
    basis = {'B': ['2s', '2p'], "N": ["2s", "2p"]}

    dftbscc = DFTBSCC(basis=basis, sk_path=sk_path, overlap=True)

    # Check default dtype attributes
    assert dftbscc.scc_dtype == torch.float64, \
        f"Default scc_dtype should be float64, got {dftbscc.scc_dtype}"
    assert dftbscc.scc_cdtype == torch.complex128, \
        f"Default scc_cdtype should be complex128, got {dftbscc.scc_cdtype}"

    # Check that model dtype matches
    assert dftbscc.model.dtype == torch.float64, \
        f"Model dtype should be float64, got {dftbscc.model.dtype}"


def test_scc_dtype_float32(rootdir=rootdir):
    """Test that scc_dtype=torch.float32 works correctly."""
    sk_path = os.path.join(rootdir, 'dftb')
    basis = {'B': ['2s', '2p'], "N": ["2s", "2p"]}

    dftbscc = DFTBSCC(basis=basis, sk_path=sk_path, overlap=True, scc_dtype=torch.float32)

    # Check dtype attributes
    assert dftbscc.scc_dtype == torch.float32, \
        f"scc_dtype should be float32, got {dftbscc.scc_dtype}"
    assert dftbscc.scc_cdtype == torch.complex64, \
        f"scc_cdtype should be complex64, got {dftbscc.scc_cdtype}"

    # Check that model dtype matches
    assert dftbscc.model.dtype == torch.float32, \
        f"Model dtype should be float32, got {dftbscc.model.dtype}"


def test_scc_dtype_float64_explicit(rootdir=rootdir):
    """Test that scc_dtype=torch.float64 can be set explicitly."""
    sk_path = os.path.join(rootdir, 'dftb')
    basis = {'B': ['2s', '2p'], "N": ["2s", "2p"]}

    dftbscc = DFTBSCC(basis=basis, sk_path=sk_path, overlap=True, scc_dtype=torch.float64)

    # Check dtype attributes
    assert dftbscc.scc_dtype == torch.float64, \
        f"scc_dtype should be float64, got {dftbscc.scc_dtype}"
    assert dftbscc.scc_cdtype == torch.complex128, \
        f"scc_cdtype should be complex128, got {dftbscc.scc_cdtype}"


def test_scc_dtype_float32_run_iters(rootdir=rootdir):
    """Test that SCC iterations work correctly with float32 precision."""
    sk_path = os.path.join(rootdir, 'dftb')
    struct = os.path.join(rootdir, 'hBN/hBN.vasp')
    basis = {'B': ['2s', '2p'], "N": ["2s", "2p"]}
    AtomicData_options = {"r_max": {'B': 6.349479778742587, 'N': 5.366822193937187}}
    kmeshgrid = [20, 20, 1]
    nel_atom = {'B': 3, 'N': 5}

    dftbscc = DFTBSCC(basis=basis, sk_path=sk_path, overlap=True, scc_dtype=torch.float32)

    dftbscc.run_iters(
        data=struct,
        nel_atom=nel_atom,
        kmeshgrid=kmeshgrid,
        kgamma_center=True,
        krotational_symmetry=False,
        ktime_inversion_symmetry=True,
        AtomicData_options=AtomicData_options,
        max_iter=1000
    )

    # Verify calculation completed and results have correct dtypes
    assert dftbscc.elec_totE is not None, "elec_totE should not be None after convergence"
    assert dftbscc.elec_H0_bandE is not None, "elec_H0_bandE should not be None after convergence"
    assert dftbscc.scc_shift_energy is not None, "scc_shift_energy should not be None after convergence"

    # Check that tensor dtypes are float32
    assert dftbscc.elec_H0_bandE.dtype == torch.float32, \
        f"elec_H0_bandE should be float32, got {dftbscc.elec_H0_bandE.dtype}"
    assert dftbscc.Gamma.dtype == torch.float32, \
        f"Gamma should be float32, got {dftbscc.Gamma.dtype}"
    assert dftbscc.scc_shift.dtype == torch.float32, \
        f"scc_shift should be float32, got {dftbscc.scc_shift.dtype}"


def test_scc_dtype_float32_vs_float64_consistency(rootdir=rootdir):
    """Test that float32 and float64 give consistent results within tolerance.

    Due to precision differences, float32 and float64 results will differ slightly,
    but they should be consistent within a reasonable tolerance.
    """
    sk_path = os.path.join(rootdir, 'dftb')
    struct = os.path.join(rootdir, 'hBN/hBN.vasp')
    basis = {'B': ['2s', '2p'], "N": ["2s", "2p"]}
    AtomicData_options = {"r_max": {'B': 6.349479778742587, 'N': 5.366822193937187}}
    kmeshgrid = [20, 20, 1]
    nel_atom = {'B': 3, 'N': 5}
    run_params = {
        'nel_atom': nel_atom,
        'kmeshgrid': kmeshgrid,
        'kgamma_center': True,
        'krotational_symmetry': False,
        'ktime_inversion_symmetry': True,
        'AtomicData_options': AtomicData_options,
        'max_iter': 1000
    }

    # Run with float64 (default)
    dftbscc_f64 = DFTBSCC(basis=basis, sk_path=sk_path, overlap=True, scc_dtype=torch.float64)
    dftbscc_f64.run_iters(data=struct, **run_params)

    # Run with float32
    dftbscc_f32 = DFTBSCC(basis=basis, sk_path=sk_path, overlap=True, scc_dtype=torch.float32)
    dftbscc_f32.run_iters(data=struct, **run_params)

    # Compare results with relaxed tolerance for single precision
    # float32 has ~7 significant digits, so we use atol=1e-4 for energy comparisons
    assert torch.allclose(
        dftbscc_f64.elec_totE.float(),
        dftbscc_f32.elec_totE,
        atol=1e-4
    ), f"elec_totE differs too much: f64={dftbscc_f64.elec_totE.item()}, f32={dftbscc_f32.elec_totE.item()}"

    assert torch.allclose(
        dftbscc_f64.elec_H0_bandE.float(),
        dftbscc_f32.elec_H0_bandE,
        atol=1e-4
    ), f"elec_H0_bandE differs too much: f64={dftbscc_f64.elec_H0_bandE.item()}, f32={dftbscc_f32.elec_H0_bandE.item()}"

    assert torch.allclose(
        dftbscc_f64.scc_shift.float(),
        dftbscc_f32.scc_shift,
        atol=1e-3
    ), f"scc_shift differs too much"

    assert np.allclose(
        dftbscc_f64.mulliken.mul_charge,
        dftbscc_f32.mulliken.mul_charge,
        atol=1e-4
    ), (
        f"mul_charge differs too much between float64 and float32 SCC runs; "
        f"max_abs_diff={np.max(np.abs(dftbscc_f64.mulliken.mul_charge - dftbscc_f32.mulliken.mul_charge))}"
    )


# ============================================================================
# Tests for smooth_ski parameter
# ============================================================================

def test_smooth_ski_initialization(rootdir=rootdir):
    """Test that DFTBSCC can be initialized with smooth_ski=True."""
    sk_path = os.path.join(rootdir, 'dftb')
    basis = {'B': ['2s', '2p'], "N": ["2s", "2p"]}

    dftbscc = DFTBSCC(basis=basis, sk_path=sk_path, overlap=True, smooth_ski=True)

    # Check that the model was initialized with smooth_intp interp method
    assert dftbscc.model is not None, "Model should be initialized"
    assert dftbscc.model.model_options['dftbsk']['interp_method'] == 'smooth_intp', \
        "interp_method should be 'smooth_intp' when smooth_ski=True"


def test_smooth_ski_hBN(rootdir=rootdir):
    """Test DFTBSCC with smooth_ski=True on hBN structure."""
    sk_path = os.path.join(rootdir, 'dftb')
    struct = os.path.join(rootdir, 'hBN/hBN.vasp')
    basis = {'B': ['2s', '2p'], "N": ["2s", "2p"]}
    AtomicData_options = {"r_max": {'B': 6.349479778742587, 'N': 5.366822193937187}}
    kmeshgrid = [20, 20, 1]
    nel_atom = {'B': 3, 'N': 5}

    dftbscc = DFTBSCC(basis=basis, sk_path=sk_path, overlap=True, smooth_ski=True)

    dftbscc.run_iters(
        data=struct,
        nel_atom=nel_atom,
        kmeshgrid=kmeshgrid,
        kgamma_center=True,
        krotational_symmetry=False,
        ktime_inversion_symmetry=True,
        AtomicData_options=AtomicData_options,
        mix_rate=0.25,
        max_iter=1000,
        smearing_method='Fermi-Dirac'
    )

    # Verify calculation completed successfully
    assert dftbscc.elec_totE is not None, "elec_totE should not be None after convergence"
    assert dftbscc.elec_H0_bandE is not None, "elec_H0_bandE should not be None after convergence"
    assert dftbscc.scc_shift_energy is not None, "scc_shift_energy should not be None after convergence"
    assert dftbscc.E_fermi is not None, "E_fermi should not be None after convergence"

    # Reference values updated after fixing k-point weight handling in get_fermi_level
    elec_totE_ref = torch.tensor([-104.5666], dtype=torch.float64)
    elec_H0_bandE_ref = torch.tensor([-104.6111], dtype=torch.float64)
    scc_shift_energy_ref = torch.tensor(0.04448110, dtype=torch.float64)
    E_fermi_ref = -3.1990918614056083
    mulliken_ref = np.array([5.225299821013989, 2.774700178986014])
    delta_charge_ref = np.array([0.22529982101398893, -0.22529982101398582])
    scc_shift_ref = torch.tensor([0.6619, 0.2670], dtype=torch.float64)
    Gamma_ref = torch.tensor([[71.8542, 68.9164],
                              [68.9164, 67.7311]], dtype=torch.float64)

    # Check energy values
    assert torch.allclose(dftbscc.elec_totE, elec_totE_ref, atol=1e-4), \
        f"elec_totE mismatch: {dftbscc.elec_totE} vs {elec_totE_ref}"
    assert torch.allclose(dftbscc.elec_H0_bandE, elec_H0_bandE_ref, atol=1e-4), \
        f"elec_H0_bandE mismatch: {dftbscc.elec_H0_bandE} vs {elec_H0_bandE_ref}"
    assert torch.allclose(dftbscc.scc_shift_energy, scc_shift_energy_ref, atol=1e-6), \
        f"scc_shift_energy mismatch: {dftbscc.scc_shift_energy} vs {scc_shift_energy_ref}"
    assert np.isclose(dftbscc.E_fermi, E_fermi_ref, atol=1e-2), \
        f"E_fermi mismatch: {dftbscc.E_fermi} vs {E_fermi_ref}"

    # Check Mulliken charges
    assert np.allclose(dftbscc.mulliken.mul_charge, mulliken_ref, atol=1e-5), \
        f"mulliken.mul_charge mismatch: {dftbscc.mulliken.mul_charge} vs {mulliken_ref}"
    assert np.allclose(dftbscc.mulliken.delta_charge, delta_charge_ref, atol=1e-5), \
        f"mulliken.delta_charge mismatch: {dftbscc.mulliken.delta_charge} vs {delta_charge_ref}"

    # Check SCC shift and Gamma matrix
    assert torch.allclose(dftbscc.scc_shift, scc_shift_ref, atol=1e-4), \
        f"scc_shift mismatch: {dftbscc.scc_shift} vs {scc_shift_ref}"
    assert torch.allclose(dftbscc.Gamma, Gamma_ref, atol=1e-4), \
        f"Gamma mismatch"

    # Verify the energy relationship holds
    assert torch.allclose(dftbscc.elec_totE, dftbscc.elec_H0_bandE + dftbscc.scc_shift_energy, atol=1e-10), \
        "elec_totE should equal elec_H0_bandE + scc_shift_energy"

    # Check that Gamma matrix is symmetric (for 2-atom system it's 2x2)
    assert dftbscc.Gamma.shape == (2, 2), f"Gamma shape should be (2, 2), got {dftbscc.Gamma.shape}"
    assert torch.allclose(dftbscc.Gamma, dftbscc.Gamma.T, atol=1e-10), \
        "Gamma matrix should be symmetric"


def test_smooth_ski_CH4(rootdir=rootdir):
    """Test DFTBSCC with smooth_ski=True on CH4 molecule."""
    sk_path = os.path.join(rootdir, 'dftb')
    struct = os.path.join(rootdir, 'ch4/CH4.vasp')
    basis = {'C': ['2s', '2p'], "H": ["1s"]}
    AtomicData_options = {"r_max": {'C': 6.2, 'H': 4.5}}
    kmeshgrid = [1, 1, 1]
    nel_atom = {'C': 4, 'H': 1}

    dftbscc = DFTBSCC(basis=basis, sk_path=sk_path, overlap=True, smooth_ski=True)

    dftbscc.run_iters(
        data=struct,
        nel_atom=nel_atom,
        kmeshgrid=kmeshgrid,
        kgamma_center=True,
        krotational_symmetry=True,
        ktime_inversion_symmetry=True,
        AtomicData_options=AtomicData_options,
        mix_rate=0.25,
        max_iter=1000,
        smearing_method='Fermi-Dirac'
    )

    # Verify calculation completed successfully
    assert dftbscc.elec_totE is not None, "elec_totE should not be None after convergence"
    assert dftbscc.elec_H0_bandE is not None, "elec_H0_bandE should not be None after convergence"
    assert dftbscc.scc_shift_energy is not None, "scc_shift_energy should not be None after convergence"
    assert dftbscc.E_fermi is not None, "E_fermi should not be None after convergence"

    # Reference values from DeePTB SCC with smooth_ski=True
    elec_totE_ref = torch.tensor([-90.7844], dtype=torch.float64)
    elec_H0_bandE_ref = torch.tensor([-90.8235], dtype=torch.float64)
    scc_shift_energy_ref = torch.tensor(0.03902681, dtype=torch.float64)
    E_fermi_ref = -2.8185704081574876
    mulliken_ref = np.array([4.37194744, 0.9221726, 0.89649146, 0.90469426, 0.90469425])
    delta_charge_ref = np.array([0.37194744, -0.0778274, -0.10350854, -0.09530574, -0.09530575])
    scc_shift_ref = torch.tensor([0.4431, 0.2590, 0.2211, 0.2293, 0.2293], dtype=torch.float64)
    Gamma_ref = torch.tensor([[8.4996, 6.9564, 7.4888, 7.3541, 7.3541],
                              [6.9564, 9.3724, 5.4857, 5.4101, 5.4101],
                              [7.4888, 5.4857, 9.3724, 6.1239, 6.1239],
                              [7.3541, 5.4101, 6.1239, 9.3724, 5.8532],
                              [7.3541, 5.4101, 6.1239, 5.8532, 9.3724]], dtype=torch.float64)

    # Check energy values
    assert torch.allclose(dftbscc.elec_totE, elec_totE_ref, atol=1e-4), \
        f"elec_totE mismatch: {dftbscc.elec_totE} vs {elec_totE_ref}"
    assert torch.allclose(dftbscc.elec_H0_bandE, elec_H0_bandE_ref, atol=1e-4), \
        f"elec_H0_bandE mismatch: {dftbscc.elec_H0_bandE} vs {elec_H0_bandE_ref}"
    assert torch.allclose(dftbscc.scc_shift_energy, scc_shift_energy_ref, atol=1e-5), \
        f"scc_shift_energy mismatch: {dftbscc.scc_shift_energy} vs {scc_shift_energy_ref}"
    assert np.isclose(dftbscc.E_fermi, E_fermi_ref, atol=1e-2), \
        f"E_fermi mismatch: {dftbscc.E_fermi} vs {E_fermi_ref}"

    # Check Mulliken charges
    assert np.allclose(dftbscc.mulliken.mul_charge, mulliken_ref, atol=5e-5), \
        f"mulliken.mul_charge mismatch: {dftbscc.mulliken.mul_charge} vs {mulliken_ref}"
    assert np.allclose(dftbscc.mulliken.delta_charge, delta_charge_ref, atol=5e-5), \
        f"mulliken.delta_charge mismatch: {dftbscc.mulliken.delta_charge} vs {delta_charge_ref}"

    # Check SCC shift and Gamma matrix
    assert torch.allclose(dftbscc.scc_shift, scc_shift_ref, atol=2e-4), \
        f"scc_shift mismatch: {dftbscc.scc_shift} vs {scc_shift_ref}"
    assert torch.allclose(dftbscc.Gamma, Gamma_ref, atol=3e-4), \
        f"Gamma mismatch"

    # Verify the energy relationship holds
    assert torch.allclose(dftbscc.elec_totE, dftbscc.elec_H0_bandE + dftbscc.scc_shift_energy, atol=1e-10), \
        "elec_totE should equal elec_H0_bandE + scc_shift_energy"

    # Gamma matrix should be 5x5 for CH4
    assert dftbscc.Gamma.shape == (5, 5), f"Gamma shape should be (5, 5), got {dftbscc.Gamma.shape}"
    assert torch.allclose(dftbscc.Gamma, dftbscc.Gamma.T, atol=1e-10), \
        "Gamma matrix should be symmetric"


def test_smooth_ski_vs_standard_mode(rootdir=rootdir):
    """Test that smooth_ski and standard mode give consistent but different results.

    Since the interpolation methods differ, the results will not be identical,
    but they should be physically reasonable and relatively close.
    """
    sk_path = os.path.join(rootdir, 'dftb')
    struct = os.path.join(rootdir, 'hBN/hBN.vasp')
    basis = {'B': ['2s', '2p'], "N": ["2s", "2p"]}
    AtomicData_options = {"r_max": {'B': 6.349479778742587, 'N': 5.366822193937187}}
    kmeshgrid = [20, 20, 1]
    nel_atom = {'B': 3, 'N': 5}
    run_params = {
        'nel_atom': nel_atom,
        'kmeshgrid': kmeshgrid,
        'kgamma_center': True,
        'krotational_symmetry': False,
        'ktime_inversion_symmetry': True,
        'AtomicData_options': AtomicData_options,
        'max_iter': 1000
    }

    # Run with standard mode (smooth_ski=False)
    dftbscc_std = DFTBSCC(basis=basis, sk_path=sk_path, overlap=True, smooth_ski=False)
    dftbscc_std.run_iters(data=struct, **run_params)

    # Run with smooth_ski=True
    dftbscc_dftb = DFTBSCC(basis=basis, sk_path=sk_path, overlap=True, smooth_ski=True)
    dftbscc_dftb.run_iters(data=struct, **run_params)

    # Both should converge successfully
    assert dftbscc_std.elec_totE is not None, "Standard mode should converge"
    assert dftbscc_dftb.elec_totE is not None, "smooth_ski mode should converge"

    # Results should be different due to different interpolation methods
    # but the difference should be physically reasonable (within ~1 eV for this system)
    energy_diff = abs(dftbscc_std.elec_totE.item() - dftbscc_dftb.elec_totE.item())
    assert energy_diff < 1.0, \
        f"Energy difference between modes should be reasonable, got {energy_diff} eV"

    # Charge distributions should be similar (within 0.1 e per atom)
    charge_diff = np.abs(dftbscc_std.mulliken.mul_charge - dftbscc_dftb.mulliken.mul_charge)
    assert np.all(charge_diff < 0.1), \
        f"Charge difference should be small, max diff: {charge_diff.max()}"


def test_smooth_ski_with_float32(rootdir=rootdir):
    """Test DFTBSCC with smooth_ski=True and float32 precision."""
    sk_path = os.path.join(rootdir, 'dftb')
    struct = os.path.join(rootdir, 'hBN/hBN.vasp')
    basis = {'B': ['2s', '2p'], "N": ["2s", "2p"]}
    AtomicData_options = {"r_max": {'B': 6.349479778742587, 'N': 5.366822193937187}}
    kmeshgrid = [20, 20, 1]
    nel_atom = {'B': 3, 'N': 5}

    dftbscc = DFTBSCC(
        basis=basis,
        sk_path=sk_path,
        overlap=True,
        smooth_ski=True,
        scc_dtype=torch.float32
    )

    # Check dtypes are correctly set
    assert dftbscc.scc_dtype == torch.float32, "scc_dtype should be float32"
    assert dftbscc.model.dtype == torch.float32, "Model dtype should be float32"

    dftbscc.run_iters(
        data=struct,
        nel_atom=nel_atom,
        kmeshgrid=kmeshgrid,
        kgamma_center=True,
        krotational_symmetry=False,
        ktime_inversion_symmetry=True,
        AtomicData_options=AtomicData_options,
        max_iter=1000
    )

    # Verify calculation completed and dtypes are correct
    assert dftbscc.elec_totE is not None, "elec_totE should not be None"
    assert dftbscc.Gamma.dtype == torch.float32, "Gamma should be float32"
    assert dftbscc.scc_shift.dtype == torch.float32, "scc_shift should be float32"


def test_smooth_ski_reuse_instance(rootdir=rootdir):
    """Test that DFTBSCC with smooth_ski=True can be reused for multiple structures."""
    sk_path = os.path.join(rootdir, 'dftb')
    struct_hBN = os.path.join(rootdir, 'hBN/hBN.vasp')
    basis_hBN = {'B': ['2s', '2p'], "N": ["2s", "2p"]}
    AtomicData_options_hBN = {"r_max": {'B': 6.349479778742587, 'N': 5.366822193937187}}
    nel_atom_hBN = {'B': 3, 'N': 5}

    dftbscc = DFTBSCC(basis=basis_hBN, sk_path=sk_path, overlap=True, smooth_ski=True)

    # First run
    dftbscc.run_iters(
        data=struct_hBN,
        nel_atom=nel_atom_hBN,
        kmeshgrid=[20, 20, 1],
        kgamma_center=True,
        krotational_symmetry=False,
        ktime_inversion_symmetry=True,
        AtomicData_options=AtomicData_options_hBN,
        max_iter=1000
    )
    elec_totE_run1 = dftbscc.elec_totE.clone()

    # Second run with same structure (should give identical results)
    dftbscc.run_iters(
        data=struct_hBN,
        nel_atom=nel_atom_hBN,
        kmeshgrid=[20, 20, 1],
        kgamma_center=True,
        krotational_symmetry=False,
        ktime_inversion_symmetry=True,
        AtomicData_options=AtomicData_options_hBN,
        max_iter=1000
    )
    elec_totE_run2 = dftbscc.elec_totE.clone()

    assert torch.allclose(elec_totE_run1, elec_totE_run2, atol=1e-10), \
        f"Reused instance should give identical results: {elec_totE_run1} vs {elec_totE_run2}"
