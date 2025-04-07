import json
from pathlib import Path
import pytest
from dptb.utils.config_sk import TrainFullConfigSK, TestFullConfigSK
from dptb.utils.config_skenv import TrainFullConfigSKEnv, TestFullConfigSKEnv
from dptb.utils.config_e3 import TrainFullConfigE3, TestFullConfigE3

from dptb.entrypoints.config import config, get_full_config

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    return str(request.config.rootdir)

def test_get_full_config_success():
    """
    测试 `get_full_config` 函数在所有有效参数组合下是否正常工作
    """
    expected_configs = {
        (None,"train", True, False, False): TrainFullConfigE3,
        (None,"train", False, True, False): TrainFullConfigSK,
        (None,"train", False, False, True): TrainFullConfigSKEnv,
        (None,"test", True, False, False): TestFullConfigE3,
        (None,"test", False, True, False): TestFullConfigSK,
        (None,"test", False, False, True): TestFullConfigSKEnv,
    }

    for (model, mode, e3tb, sktb, sktbenv), expected_config in expected_configs.items():
        name, full_config = get_full_config(
                model,
                mode.lower() == "train",
                mode.lower() == "test",
                e3tb,
                sktb,
                sktbenv,
            )

        assert name == f"{mode}_{'E3' if e3tb else 'SK' if sktb else 'SKEnv'}"
        assert full_config == expected_config


def test_get_full_config_errors():
    """
    测试 `get_full_config` 函数是否在无效参数下抛出异常
    """
    with pytest.raises(ValueError) as excinfo:
        get_full_config(None,False, False, False, False, False)
    assert "Unknown mode" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        get_full_config(None,True, False, False, False, False)
    assert "Unknown config type" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        get_full_config(None,False, True, False, False, False)
    assert "Unknown config type" in str(excinfo.value)


def test_config_success(root_directory):
    """
    测试 `config` 函数在所有有效参数组合下是否正常工作
    """
    expected_configs = {
        ("train", True, False, False): TrainFullConfigE3,
        ("train", False, True, False): TrainFullConfigSK,
        ("train", False, False, True): TrainFullConfigSKEnv,
        ("test", True, False, False): TestFullConfigE3,
        ("test", False, True, False): TestFullConfigSK,
        ("test", False, False, True): TestFullConfigSKEnv,
    }
    tmp_path = f'{root_directory}/dptb/tests/data/test_sktb/output'
    for (mode, e3tb, sktb, sktbenv), expected_config in expected_configs.items():
        path =f"{tmp_path}/config_{mode}_{'E3' if e3tb else 'SK' if sktb else 'SKEnv'}.json"

        # if tmp_path not exist, create it
        if not Path(tmp_path).exists():
            Path(tmp_path).mkdir(parents=True, exist_ok=True)
            
        config(
                str(path),
                mode.lower() == "train",
                mode.lower() == "test",
                e3tb,
                sktb,
                sktbenv,
            )

        with open(path,"r") as fp:
            actual_config = json.load(fp)

        assert actual_config == expected_config


def test_config_errors(root_directory):
    """
    测试 `config` 函数是否在无效参数下抛出异常
    """
    tmp_path = root_directory + '/dptb/tests/data/test_sktb/output'
    with pytest.raises(ValueError) as excinfo:
        path = Path(tmp_path) / "config.json"
        config(str(path))
    assert "Please specify the type of config you want to generate" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        path = Path(tmp_path) / "config.json"
        config(str(path), False, False, False, False, False)
    assert "Please specify the type of config you want to generate" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        path = Path(tmp_path) / "config.json"
        config(str(path), False, False, True, True, False)
    assert "Please specify only one of e3tb, sktb, sktbenv" in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        path = Path(tmp_path) / "config.json"
        config(str(path), False, False, True, True, True)
    assert "Please specify only one of e3tb, sktb, sktbenv" in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        path = Path(tmp_path) / "config.json"
        config(str(path), False, False, True, False, True)
    assert "Please specify only one of e3tb, sktb, sktbenv" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        path = Path(tmp_path) / "config.json"
        config(str(path), True, True, True, False, False)
    assert "Please specify only one of train and test" in str(excinfo.value)