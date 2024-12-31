import pytest
from dptb.data.build import build_dataset
from dptb.data.dataset import DefaultDataset
from dptb.data.transforms import OrbitalMapper

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    return str(request.config.rootdir)


def test_build_dataset_success(root_directory):
    set_options = {
        "r_max": 5.0,
        "er_max": 5.0,
        "oer_max": 2.5,
        "root": f"{root_directory}/dptb/tests/data/test_sktb/dataset",
        "prefix": "kpath_spk",
        "get_eigenvalues": True,
        "get_Hamiltonian": False,
    }
    common_options={"basis": {"Si": ["3s", "3p"]}}


    dataset = build_dataset(**set_options, **common_options)

    # Assert that the dataset is of the expected type
    assert isinstance(dataset, DefaultDataset)

    # Assert that the dataset root is set correctly
    assert dataset.root == f"{root_directory}/dptb/tests/data/test_sktb/dataset"

    # Assert that the dataset type_mapper is set correctly
    assert isinstance (dataset.type_mapper,OrbitalMapper)

    # Assert that the dataset get_Hamiltonian option is set correctly
    assert dataset.get_Hamiltonian == False

    # Assert that the dataset get_eigenvalues option is set correctly
    assert dataset.get_eigenvalues == True

    # Assert that the dataset info_files is populated correctly
    assert len(dataset.info_files) > 0

    # Assert that the dataset info_files contains the expected keys
    assert "kpath_spk.0" in dataset.info_files

    # Assert that the dataset info_files values are of the expected type
    assert isinstance(dataset.info_files["kpath_spk.0"], dict)

    # Assert that the dataset info_files values have the expected keys
    assert "bandinfo" in dataset.info_files['kpath_spk.0']

    assert isinstance(dataset.info_files["kpath_spk.0"]["bandinfo"], dict)

def test_build_dataset_rmax_dict(root_directory):
    set_options = {
        "r_max": {'Si':5.0},
        "er_max": 5.0,
        "oer_max": 2.5,
        "root": f"{root_directory}/dptb/tests/data/test_sktb/dataset",
        "prefix": "kpath_spk",
        "get_eigenvalues": True,
        "get_Hamiltonian": False,
    }
    common_options={"basis": {"Si": ["3s", "3p"]}}


    dataset = build_dataset(**set_options, **common_options)

def test_build_dataset_rmax_dict_bondwise(root_directory):
    set_options = {
        "r_max": {'Si-Si':5.0},
        "er_max": 5.0,
        "oer_max": 2.5,
        "root": f"{root_directory}/dptb/tests/data/test_sktb/dataset",
        "prefix": "kpath_spk",
        "get_eigenvalues": True,
        "get_Hamiltonian": False,
    }
    common_options={"basis": {"Si": ["3s", "3p"]}}
    
    dataset = build_dataset(**set_options, **common_options)


def test_build_dataset_fail(root_directory):
    set_options = {
        "r_max": 5.0,
        "er_max": 5.0,
        "oer_max": 2.5,
        "root": f"{root_directory}/dptb/tests/data/test_sktb/dataset",
        "prefix": "kpath_spk",
        "get_eigenvalues": False,
        "get_Hamiltonian": True,
    }
    common_options={"basis": {"Si": ["3s", "3p"]}}

    with pytest.raises(AssertionError) as excinfo:
        dataset = build_dataset(**set_options, **common_options)
    assert "Hamiltonian file not found" in str(excinfo.value)



#TODO: Add failure test cases for build_dataset. when get_eigenvalues is True and get_Hamiltonian is False; 当我们补充E3的测试案例时，会有一个数据集，只有Hamiltonian，没有eigenvalues。我们需要测试这种情况。