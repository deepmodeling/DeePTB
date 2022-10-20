from dptb.entrypoints.train import train
from dptb.entrypoints.main import parse_args
import os
from pathlib import Path
import logging

INPUT = os.path.join(Path(os.path.abspath(__file__)).parent, "data/input.json")
INPUT_nnsk = os.path.join(Path(os.path.abspath(__file__)).parent, "data/input_nnsk.json")
INPUT_adding_orbital = os.path.join(Path(os.path.abspath(__file__)).parent, "data/input_nnsk_d.json")

test_data_path = os.path.join(Path(os.path.abspath(__file__)).parent, "data/")



log = logging.getLogger(__name__)

def test_train():
    train(
        INPUT = INPUT,
        init_model = None,
        restart=None,
        freeze=False,
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
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        train_sk=True,
        use_correction=False,
    )

def test_train_init_model():
    train(
        INPUT=INPUT,
        init_model=test_data_path+"/hBN/checkpoint/best_dptb.pth",
        restart=None,
        freeze=False,
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
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        train_sk=False,
        use_correction=test_data_path+"/hBN/checkpoint/best_nnsk.pth",
    )

def test_train_init_model_crt():
    train(
        INPUT=INPUT,
        init_model=test_data_path+"/hBN/checkpoint/best_dptb.pth",
        restart=None,
        freeze=False,
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
        output=test_data_path+"/test_all/fancy_ones",
        log_level=2,
        log_path=None,
        train_sk=True,
        use_correction=None,
    )