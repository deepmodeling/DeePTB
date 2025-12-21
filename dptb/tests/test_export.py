import pytest
import os
import numpy as np
from dptb.data import AtomicData
from dptb.nn import build_model
from ase.io import read
from dptb.postprocess.interfaces import ToWannier90, ToPythTB

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    return str(request.config.rootdir)

@pytest.fixture
def model_and_data(root_directory):
    # Use output from test_sktb.py
    # Assumes test_sktb.py runs first. We use ordering to ensure this.
    model_path = root_directory+"/dptb/tests/data/test_sktb/output/test_valence/checkpoint/nnsk.latest.pth"
    
    # Check if model exists (sanity check if run in isolation without previous tests)
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found at {model_path}. Run test_sktb.py first.")

    # Load model
    # Note: build_model usually takes config options, but if loading from checkpoint, 
    # we might need empty options or it mimics CLI behavior?
    # CLI uses dptb.nn.build.build_model(chkpt, options, ...)
    # If options are empty, it might fail if model needs them?
    # nnsk.latest.pth typically contains full config.
    
    # We construct minimal options required by build_model if needed
    # But usually build_model can load from pth if provided as first arg?
    # Actually build_model signature: build_model(model_name_or_path, model_options, common_options)
    
    # We define minimal common_options
    common_options = {
        "device": "cpu",
        "dtype": "float32",
        "overlap": False, 
    }
    
    model = build_model(model_path, {}, common_options)
    model.to("cpu")
    model.eval()
    
    # Structure file
    # test_valence used Si structure. We use silicon.vasp which should be compatible.
    struc_file = root_directory+"/dptb/tests/data/silicon_1nn/silicon.vasp"
    
    return model, struc_file

@pytest.mark.order(2) # ensure it runs after test_sktb (order 1)
def test_wannier90_export(model_and_data, tmp_path):
    model, struc_file = model_and_data
    exporter = ToWannier90(model=model, device="cpu")
    
    hr_file = tmp_path / "test_hr.dat"
    win_file = tmp_path / "test.win"
    cen_file = tmp_path / "test_centres.xyz"
    
    # Run export methods using FILE PATH string
    # This verifies load_data_for_model handles string input correctly
    # and extracts r_max from model options automatically.
    exporter.write_hr(struc_file, str(hr_file), e_fermi=-7.72)
    exporter.write_win(struc_file, str(win_file), e_fermi=-7.72)
    exporter.write_centres(struc_file, str(cen_file))
    
    # Verify files exist and have content
    assert hr_file.exists()
    assert win_file.exists()
    assert cen_file.exists()
    
    # Simple content checks
    with open(hr_file) as f:
        lines = f.readlines()
        assert "written by DeePTB" in lines[0]
        # num_bands check
        # Si basis in input_valence.json: ["3s", "3p"] -> 4 orbitals/atom.
        # 2 atoms -> 8 bands.
        assert lines[1].strip() == "8" 

@pytest.mark.order(2)
def test_pythtb_export(model_and_data):
    try:
        import pythtb
    except ImportError:
        pytest.skip("pythtb not installed")
        
    model, struc_file = model_and_data
    exporter = ToPythTB(model=model, device="cpu")
    
    # Convert to PythTB model
    tb_model = exporter.get_model(struc_file, e_fermi=-7.72)
    
    assert tb_model is not None
    assert tb_model._norb == 8
    # Verify coordinates are fractional (should be within roughly [0, 1] for this unit cell)
    # Silicon positions are 0.0 and 0.25. If Cartesian, they would be > 1.0 (lattice const ~5.43, 0.25*5.43 > 1.3)
    orb_coords = tb_model.get_orb()
    import numpy as np
    assert np.all(orb_coords >= -0.01) and np.all(orb_coords <= 1.01), f"Coordinates mismatch: {orb_coords}"

@pytest.mark.order(3)
def test_overlap_error():
    # Mock a model with overlap=True
    from unittest.mock import MagicMock
    mock_model = MagicMock()
    mock_model.overlap = True # Simulate overlap model
    mock_model.idp = MagicMock() # Needs idp for init usually? No, init doesn't access idp, methods do.
    
    # Check ToWannier90
    exporter_w90 = ToWannier90(model=mock_model)
    with pytest.raises(ValueError, match="overlap"):
        exporter_w90._get_data_and_blocks(data="dummy_path")
        
    # Check ToPythTB
    exporter_pythtb = ToPythTB(model=mock_model)
    with pytest.raises(ValueError, match="overlap"):
        exporter_pythtb.get_model(data="dummy_path")
