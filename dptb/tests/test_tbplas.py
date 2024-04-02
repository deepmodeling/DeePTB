import pytest
from dptb.data import AtomicData
from dptb.nn import build_model
from ase.io import read


@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    return str(request.config.rootdir)

@pytest.mark.order(1)
def test_tbplas_get_cell(root_directory):
    try:
        import tbplas
    except:
        pytest.skip("TBPLaS is not installed in the current image, please check the Dockerfile of the workflow.")
    from dptb.postprocess.totbplas import TBPLaS
    common_options = {
        "basis": {
            "Si": ["3s", "3p", "d*"],
        },
        "device": "cpu",
        "dtype": "float32",
        "overlap": False,
    }

    run_opt = {
            "init_model": root_directory+"/dptb/tests/data/silicon_1nn/nnsk.ep500.pth",
            "restart": None,
            "freeze": False,
            "train_soc": False,
            "log_path": None,
            "log_level": None
        }

    dataset = AtomicData.from_ase(
        atoms=read(root_directory+"/dptb/tests/data/silicon_1nn/silicon.vasp"),
        r_max=3.0,
        er_max=3.0,
        oer_max=2.5,
        )


    model = build_model(run_opt["init_model"], {}, common_options)
    model.to(common_options["device"])

    model.eval()
    tbplas = TBPLaS(model=model, device="cpu")
    cell = tbplas.get_cell(data=dataset, e_fermi=-7.724611085233356)

    # check cell onsite value
    assert abs(cell.get_orbital(0).energy + 5.0404769961265075) < 1e-5
    assert abs(cell.get_orbital(1).energy - 3.4503467496010316) < 1e-5
    assert abs(cell.get_orbital(-1).energy - 11.223119538602496) < 1e-5

    # check cell hopping value
    assert abs(cell.get_hopping(orb_i=0, orb_j=9, rn=(0,0,0)) - 1.8239758014678955) < 1e-5
    assert abs(cell.get_hopping(orb_i=0, orb_j=10, rn=(0,0,0)) - 1.259192943572998) < 1e-5
