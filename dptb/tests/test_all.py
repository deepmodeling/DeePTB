from dptb.entrypoints.train import train
from dptb.entrypoints.main import parse_args
import os
from pathlib import Path
from dptb.plugins.train_logger import Logger
from dptb.nnops.train_dptb import DPTBTrainer
from dptb.nnops.plugins import Saver
from dptb.nnops.train_nnsk import NNSKTrainer
from dptb.nnops.monitor import TrainLossMonitor, LearningRateMonitor, Validationer
import time
import json
import logging
from dptb.utils.tools import j_loader
from dptb.utils.loggers import set_log_handles
from typing import Dict, List, Optional, Any


INPUT = os.path.join(Path(os.path.abspath(__file__)).parent, "data/input.json")
test_data_path = os.path.join(Path(os.path.abspath(__file__)).parent, "data/")



log = logging.getLogger(__name__)

def _test_train():
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
        INPUT=INPUT,
        init_model=None,
        restart=None,
        freeze=False,
        output=test_data_path+"/test_all",
        log_level=2,
        log_path=None,
        train_sk=True,
        use_correction=False,
    )

def _test_train_init_model():
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

def _test_train_sk_init_model():
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


def _test_train_crt():
    init_model = None
    restart = None
    freeze = False
    output = test_data_path + "/test_all"
    log_level = 2
    log_path = None
    train_sk = False
    use_correction = True

    run_opt = {
        "init_model": init_model,
        "restart": restart,
        "freeze": freeze,
        "log_path": log_path,
        "log_level": log_level,
        "train_sk": train_sk,
        "use_correction": use_correction
    }

    if all((init_model, restart)):
        raise RuntimeError(
            "--init-model and --restart should not be set at the same time"
        )
    if all((use_correction, train_sk)):
        raise RuntimeError(
            "--use-cprrection and --train_sk should not be set at the same time"
        )

    # setup INPUT path
    if train_sk:
        if init_model:
            skconfig_path = os.path.join(str(Path(init_model).parent.absolute()), "config_nnsktb.json")
            mode = "init_model"
        elif restart:
            skconfig_path = os.path.join(str(Path(restart).parent.absolute()), "config_nnsktb.json")
            mode = "restart"
        elif INPUT is not None:
            mode = "from_scratch"
            skconfig_path = INPUT
        else:
            log.error("ValueError: Missing Input configuration file path.")
            raise ValueError

    else:
        if init_model:
            dptbconfig_path = os.path.join(str(Path(init_model).parent.absolute()), "config_dptbtb.json")
            mode = "init_model"
        elif restart:
            dptbconfig_path = os.path.join(str(Path(restart).parent.absolute()), "config_dptbtb.json")
            mode = "restart"
        elif INPUT is not None:
            dptbconfig_path = INPUT
            mode = "from_scratch"
        else:
            log.error("ValueError: Missing Input configuration file path.")
            raise ValueError

        if use_correction:
            skcheckpoint_path = test_data_path + "/hBN/checkpoint/best_nnsk.pth"

    # setup output path
    if output:
        Path(output).parent.mkdir(exist_ok=True, parents=True)
        Path(output).mkdir(exist_ok=True, parents=True)
        checkpoint_path = os.path.join(str(output), "checkpoint")
        Path(checkpoint_path).mkdir(exist_ok=True, parents=True)
        if not log_path:
            log_path = os.path.join(str(output), "log/log.txt")
        Path(log_path).parent.mkdir(exist_ok=True, parents=True)

        # parse training configuration, if INPUT is None and restart or init model is True, we can load the configure of
        # checkpoint
        run_opt.update({
            "output": str(Path(output).absolute()),
            "checkpoint_path": str(Path(checkpoint_path).absolute()),
            "log_path": str(Path(log_path).absolute())
        })

    run_opt.update({"mode": mode})
    if train_sk:
        run_opt.update({
            "skconfig_path": skconfig_path,
        })
    else:
        if use_correction:
            run_opt.update({
                "skcheckpoint_path": skcheckpoint_path
            })
        run_opt.update({
            "dptbconfig_path": dptbconfig_path
        })

    set_log_handles(log_level, Path(log_path) if log_path else None)
    # parse the config. Since if use init, config file may not equals to current
    jdata = j_loader(INPUT)
    if train_sk:
        trainer = NNSKTrainer(run_opt, jdata)
    else:
        trainer = DPTBTrainer(run_opt, jdata)

    # register the plugin in trainer, to tract training info
    trainer.register_plugin(Validationer())
    trainer.register_plugin(TrainLossMonitor())
    trainer.register_plugin(LearningRateMonitor())
    trainer.register_plugin(Logger(["train_loss", "validation_loss", "lr"]))

    if output:
        # output training configurations:
        with open(os.path.join(output, "train_config.json"), "w") as fp:
            json.dump(jdata, fp, indent=4)

        trainer.register_plugin(Saver(
            interval=[(jdata["train_options"].get("save_epoch"), 'epoch')] if jdata["train_options"].get(
                "save_epoch") else None))
        # add a plugin to save the training parameters of the model, with model_output as given path

    start_time = time.time()

    trainer.run(trainer.num_epoch)

    end_time = time.time()
    log.info("finished training")
    log.info(f"wall time: {(end_time - start_time):.3f} s")

def _test_train_init_model_crt():
    init_model = test_data_path+"/hBN/checkpoint/best_dptb.pth"
    restart = None
    freeze = False
    output = test_data_path+"/test_all"
    log_level = 2
    log_path = None
    train_sk = False
    use_correction = True

    run_opt = {
        "init_model": init_model,
        "restart": restart,
        "freeze": freeze,
        "log_path": log_path,
        "log_level": log_level,
        "train_sk": train_sk,
        "use_correction": use_correction
    }

    if all((init_model, restart)):
        raise RuntimeError(
            "--init-model and --restart should not be set at the same time"
        )
    if all((use_correction, train_sk)):
        raise RuntimeError(
            "--use-cprrection and --train_sk should not be set at the same time"
        )

    # setup INPUT path
    if train_sk:
        if init_model:
            skconfig_path = os.path.join(str(Path(init_model).parent.absolute()), "config_nnsktb.json")
            mode = "init_model"
        elif restart:
            skconfig_path = os.path.join(str(Path(restart).parent.absolute()), "config_nnsktb.json")
            mode = "restart"
        elif INPUT is not None:
            mode = "from_scratch"
            skconfig_path = INPUT
        else:
            log.error("ValueError: Missing Input configuration file path.")
            raise ValueError

    else:
        if init_model:
            dptbconfig_path = os.path.join(str(Path(init_model).parent.absolute()), "config_dptbtb.json")
            mode = "init_model"
        elif restart:
            dptbconfig_path = os.path.join(str(Path(restart).parent.absolute()), "config_dptbtb.json")
            mode = "restart"
        elif INPUT is not None:
            dptbconfig_path = INPUT
            mode = "from_scratch"
        else:
            log.error("ValueError: Missing Input configuration file path.")
            raise ValueError

        if use_correction:
            skcheckpoint_path = test_data_path+"/hBN/checkpoint/best_nnsk.pth"

    # setup output path
    if output:
        Path(output).parent.mkdir(exist_ok=True, parents=True)
        Path(output).mkdir(exist_ok=True, parents=True)
        checkpoint_path = os.path.join(str(output), "checkpoint")
        Path(checkpoint_path).mkdir(exist_ok=True, parents=True)
        if not log_path:
            log_path = os.path.join(str(output), "log/log.txt")
        Path(log_path).parent.mkdir(exist_ok=True, parents=True)

    # parse training configuration, if INPUT is None and restart or init model is True, we can load the configure of
    # checkpoint
        run_opt.update({
            "output": str(Path(output).absolute()),
            "checkpoint_path": str(Path(checkpoint_path).absolute()),
            "log_path": str(Path(log_path).absolute())
        })

    run_opt.update({"mode": mode})
    if train_sk:
        run_opt.update({
            "skconfig_path": skconfig_path,
        })
    else:
        if use_correction:
            run_opt.update({
                "skcheckpoint_path": skcheckpoint_path
            })
        run_opt.update({
            "dptbconfig_path": dptbconfig_path
        })

    set_log_handles(log_level, Path(log_path) if log_path else None)
    # parse the config. Since if use init, config file may not equals to current
    jdata = j_loader(INPUT)
    if train_sk:
        trainer = NNSKTrainer(run_opt, jdata)
    else:
        trainer = DPTBTrainer(run_opt, jdata)

    # register the plugin in trainer, to tract training info
    trainer.register_plugin(Validationer())
    trainer.register_plugin(TrainLossMonitor())
    trainer.register_plugin(LearningRateMonitor())
    trainer.register_plugin(Logger(["train_loss", "validation_loss", "lr"]))

    if output:
        # output training configurations:
        with open(os.path.join(output, "train_config.json"), "w") as fp:
            json.dump(jdata, fp, indent=4)

        trainer.register_plugin(Saver(
            interval=[(jdata["train_options"].get("save_epoch"), 'epoch')] if jdata["train_options"].get(
                "save_epoch") else None))
        # add a plugin to save the training parameters of the model, with model_output as given path

    start_time = time.time()

    trainer.run(trainer.num_epoch)

    end_time = time.time()
    log.info("finished training")
    log.info(f"wall time: {(end_time - start_time):.3f} s")