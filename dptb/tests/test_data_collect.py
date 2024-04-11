import pytest
import os

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    return str(request.config.rootdir)

def test_data_collect(root_directory):
    from dptb.entrypoints.data import data
    INPUT = root_directory + "/dptb/tests/data/collect_config.json"

    # clear possible left-behind output
    if os.path.exists(root_directory + "/dptb/tests/data/fake_dataset_split/full/"):
        os.system("rm -r " + root_directory + "/dptb/tests/data/fake_dataset_split/full")

    data(INPUT=INPUT, collect=True)

    assert os.path.exists(root_directory + "/dptb/tests/data/fake_dataset_split/full/full.0")
    assert os.path.exists(root_directory + "/dptb/tests/data/fake_dataset_split/full/full.1")
    assert os.path.exists(root_directory + "/dptb/tests/data/fake_dataset_split/full/full.2")

    os.system("rm -r " + root_directory + "/dptb/tests/data/fake_dataset_split/full")