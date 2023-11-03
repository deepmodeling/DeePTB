from dptb.entrypoints.train import train
from dptb.entrypoints.test import _test
from dptb.entrypoints.main import parse_args
import os
from pathlib import Path
import logging

INPUT = os.path.join(Path(os.path.abspath(__file__)).parent, "data/input.json")
INPUT_nnsk = os.path.join(Path(os.path.abspath(__file__)).parent, "data/input_nnsk.json")
INPUT_adding_orbital = os.path.join(Path(os.path.abspath(__file__)).parent, "data/input_nnsk_d.json")

INPUT_nnsk_test = os.path.join(Path(os.path.abspath(__file__)).parent, "data/input_nnsk_test.json")
INPUT_dptb_test = os.path.join(Path(os.path.abspath(__file__)).parent, "data/input_dptb_test.json")

test_data_path = os.path.join(Path(os.path.abspath(__file__)).parent, "data/")

INPUT_nnsk_nrl = os.path.join(Path(os.path.abspath(__file__)).parent, "data/nrl/input_nrl.json")
INPUT_nnsk_nrl_test = os.path.join(Path(os.path.abspath(__file__)).parent, "data/nrl/input_nrl_test.json")
ckpt_nnsk_nrl_path = os.path.join(Path(os.path.abspath(__file__)).parent, "../../examples/NRL-TB/silicon/ckpt")

INPUT_nnsk_wan = os.path.join(Path(os.path.abspath(__file__)).parent, "data/wan/input_wan.json")



log = logging.getLogger(__name__)

def test_train():
    train(
        INPUT = INPUT,
        init_model = None,
        restart=None,
        freeze=False,
        train_soc=False,
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        train_sk=False,
        use_correction=False,
    )

def test_train_sk():
    print("Here",INPUT)
    train(
        INPUT=INPUT_nnsk,
        init_model=None,
        restart=None,
        freeze=False,
        train_soc=False,
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        train_sk=True,
        use_correction=False,
    )

def test_train_wan():
    train(
        INPUT = INPUT_nnsk_wan,
        init_model = None,
        restart=None,
        freeze=False,
        train_soc=False,
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        train_sk=False,
        use_correction=False,
    )

def test_train_init_model():
    train(
        INPUT=INPUT,
        init_model=test_data_path+"/hBN/checkpoint/best_dptb.pth",
        restart=None,
        freeze=False,
        train_soc=False,
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        train_sk=False,
        use_correction=False,
    )


def test_train_restart_model():
    train(
        INPUT=INPUT,
        init_model=None,
        restart=test_data_path+"/hBN/checkpoint/best_dptb.pth",
        freeze=False,
        train_soc=False,
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        train_sk=False,
        use_correction=False,
    )


def test_train_sk_init_model():
    train(
        INPUT=INPUT,
        init_model=test_data_path+"/hBN/checkpoint/best_nnsk.pth",
        restart=None,
        freeze=False,
        train_soc=False,
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        train_sk=True,
        use_correction=False,
    )

def test_train_sk_init_model_nrl():
    train(
        INPUT=INPUT_nnsk_nrl,
        init_model=ckpt_nnsk_nrl_path+"/nrl_ckpt.pth",
        restart=None,
        freeze=False,
        train_soc=False,
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        train_sk=True,
        use_correction=False,
    )

def test_train_sk_init_model_nrl_json():
    train(
        INPUT=INPUT_nnsk_nrl,
        init_model=ckpt_nnsk_nrl_path+"/nrl_ckpt.json",
        restart=None,
        freeze=False,
        train_soc=False,
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        train_sk=True,
        use_correction=False,
    )
    

def test_train_sk_restart_model():
    train(
        INPUT=INPUT,
        init_model=None,
        restart=test_data_path+"/hBN/checkpoint/best_nnsk.pth",
        freeze=False,
        train_soc=False,
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        train_sk=True,
        use_correction=False,
    )

def test_train_sk_restart_model_nrl():
    train(
        INPUT=INPUT_nnsk_nrl,
        init_model=None,
        restart=ckpt_nnsk_nrl_path+"/nrl_ckpt.pth",
        freeze=False,
        train_soc=False,
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        train_sk=True,
        use_correction=False,
    )


def test_train_crt():

    train(
        INPUT=INPUT,
        init_model=None,
        restart=None,
        freeze=False,
        train_soc=False,
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        train_sk=False,
        use_correction=test_data_path+"/hBN/checkpoint/best_nnsk.pth",
    )

def test_train_crt_nrl():

    train(
        INPUT=INPUT_nnsk_nrl,
        init_model=None,
        restart=None,
        freeze=False,
        train_soc=False,
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        train_sk=False,
        use_correction=ckpt_nnsk_nrl_path+"/nrl_ckpt.pth",
    )

def test_train_crt_nrl_json():

    train(
        INPUT=INPUT_nnsk_nrl,
        init_model=None,
        restart=None,
        freeze=False,
        train_soc=False,
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        train_sk=False,
        use_correction=ckpt_nnsk_nrl_path+"/nrl_ckpt.json",
    )


def test_train_init_model_crt():
    train(
        INPUT=INPUT,
        init_model=test_data_path+"/hBN/checkpoint/best_dptb.pth",
        restart=None,
        freeze=False,
        train_soc=False,
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        train_sk=False,
        use_correction=test_data_path+"/hBN/checkpoint/best_nnsk.pth",
    )

## the following is some fancy test of initialization method of nnsk

def test_train_nnsk_adding_orbital():
    train(
        INPUT=INPUT_adding_orbital,
        init_model=test_data_path+"/hBN/checkpoint/best_nnsk.pth",
        restart=None,
        freeze=False,
        train_soc=False,
        output=test_data_path+"/test_all/fancy_ones",
        log_level=2,
        log_path=None,
        train_sk=True,
        use_correction=None,
    )

def test_tester_nnsk():
    _test(
        INPUT=INPUT_nnsk_test,
        init_model=test_data_path+"/hBN/checkpoint/best_nnsk.pth",
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        test_sk=True,
        use_correction=False,
    )

def test_tester_nnsk_nrl():
    _test(
        INPUT=INPUT_nnsk_nrl_test,
        init_model=ckpt_nnsk_nrl_path+"/nrl_ckpt.pth",
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        test_sk=True,
        use_correction=False,
    )

def test_tester_nnsk_nrl_json():
    _test(
        INPUT=INPUT_nnsk_nrl_test,
        init_model=ckpt_nnsk_nrl_path+"/nrl_ckpt.json",
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        test_sk=True,
        use_correction=False,
    )

def test_tester_dptb():
    _test(
        INPUT=INPUT_dptb_test,
        init_model=test_data_path+"/hBN/checkpoint/best_dptb.pth",
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        test_sk=False,
        use_correction=test_data_path+"/hBN/checkpoint/best_nnsk.pth",
    )