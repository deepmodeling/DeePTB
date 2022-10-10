from dptb.nnops.train_dptb import DPTBTrainer
from dptb.nnops.train_nnsk import NNSKTrainer
from dptb.nnops.monitor import TrainLossMonitor, LearningRateMonitor, Validationer
from dptb.plugins.train_logger import Logger
from dptb.plugins.plugins import Saver
from typing import Dict, List, Optional, Any
from dptb.utils.tools import j_loader
from dptb.utils.loggers import set_log_handles
import logging
from pathlib import Path
import json
import os
import time

__all__ = ["train"]

log = logging.getLogger(__name__)


def train(
        INPUT: str,
        init_model: Optional[str],
        restart: Optional[str],
        freeze,
        output: str,
        log_level: int,
        log_path: Optional[str],
        train_sk: bool,
        use_correction: bool,
        **kwargs
):
    run_opt = {
        "init_model": init_model,
        "restart": restart,
        "freeze": freeze,
        "log_path": log_path,
        "log_level": log_level,
        "train_sk": train_sk,
        "use_correction": use_correction
    }
    '''
        -1- set up input and output directories
            noticed that, the checkpoint of sktb and dptb should be in different directory, and in train_dptb,
            there should be a workflow to load correction model from nnsktb.
        -2- parse configuration file and start training
            
    output directories has following structure:
        - ./output/
            - checkpoint/
                - latest_dptb.pth
                - best_dptb.pth
                - latest_nnsk.pth
                ...
            - log/
                - log.log
            - config_nnsktb.json
            - config_dptb.json
    '''
    # TODO: The condition switch to adjust for different run options is too complex here. This would be hard to proceed
    #  if we add more orders. So we should seperate it in the future.

    # init all paths
    # if init_model, restart or init_frez, findout the input configure file
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
            skcheckpoint_path = str(Path(str(input(f"Enter skcheckpoint_path (default ./checkpoint/best_nnsk.pth): \n"))).absolute())

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