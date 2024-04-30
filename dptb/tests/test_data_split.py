import pytest
import os

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    return str(request.config.rootdir)


def test_data_split(root_directory):
    from dptb.entrypoints.data import data
    INPUT = root_directory + "/dptb/tests/data/split_config.json"

    data(INPUT=INPUT, parse=False, split=True)

    assert os.path.exists(root_directory + "/dptb/tests/data/fake_dataset/train")
    assert os.path.exists(root_directory + "/dptb/tests/data/fake_dataset/test")
    assert os.path.exists(root_directory + "/dptb/tests/data/fake_dataset/val")

    assert len(os.listdir(root_directory + "/dptb/tests/data/fake_dataset/train")) == 4
    assert len(os.listdir(root_directory + "/dptb/tests/data/fake_dataset/test")) == 1
    assert len(os.listdir(root_directory + "/dptb/tests/data/fake_dataset/val")) == 2

    # assert os.path.exists(root_directory + "/dptb/tests/data/fake_dataset/train/frame.1")
    # assert os.path.exists(root_directory + "/dptb/tests/data/fake_dataset/train/frame.2")
    # assert os.path.exists(root_directory + "/dptb/tests/data/fake_dataset/train/frame.0")
    # assert os.path.exists(root_directory + "/dptb/tests/data/fake_dataset/train/frame.5")

    # assert os.path.exists(root_directory + "/dptb/tests/data/fake_dataset/test/frame.4")
    # assert os.path.exists(root_directory + "/dptb/tests/data/fake_dataset/test/frame.6")

    # assert os.path.exists(root_directory + "/dptb/tests/data/fake_dataset/val/frame.3")

    os.system("rm -r " + root_directory + "/dptb/tests/data/fake_dataset/train")
    os.system("rm -r " + root_directory + "/dptb/tests/data/fake_dataset/val")
    os.system("rm -r " + root_directory + "/dptb/tests/data/fake_dataset/test")


