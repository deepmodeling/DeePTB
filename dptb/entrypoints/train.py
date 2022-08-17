from dptb.nnops.train_dptb import DPTBTrainer
from dptb.nnops.train_nnsk import NNSKTrainer
from dptb.nnops.monitor import TrainLossMonitor, LearningRateMonitor, Validationer
from dptb.nnops.train_logger import Logger
from dptb.nnops.plugins import Saver
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
        init_frz_model,
        output: str,
        log_level: int,
        log_path: Optional[str],
        train_sk: bool,
        **kwargs
):
    run_opt = {
        "init_model": init_model,
        "restart": restart,
        "init_frz_model": init_frz_model,
        "log_path": log_path,
        "log_level": log_level
    }
    # TODO: The condition switch to adjust for different run options is too complex here. This would be hard to proceed
    #  if we add more orders. So we should seperate it in the future.
    # init all paths
    if all((init_model, restart)):
        raise RuntimeError(
            "--init-model and --restart should not be set at the same time"
        )

    ''' Create the output directory, contains checkpoint, logfile '''
    if output:
        Path(output).mkdir(exist_ok=True, parents=True)
        Path(output).parent.mkdir(exist_ok=True, parents=True)
        checkpoint_path = os.path.join(str(output), "checkpoint")
        Path(checkpoint_path).mkdir(exist_ok=True, parents=True)
        if not log_path:
            log_path = os.path.join(str(output), "log/log.txt")

    # load training configuration, if INPUT is None and restart or init model is True, we can load the configure of
    # checkpoint

    training_config_path = None
    if init_model:
        training_config_path = os.path.join(str(Path(init_model).parent.absolute()), "train_config.json")
    elif restart:
        training_config_path = os.path.join(str(Path(restart).parent.absolute()), "train_config.json")

    if INPUT is None and training_config_path is not None:
        INPUT = training_config_path
    elif INPUT is None and training_config_path is None:
        log.error("ValueError: Missing Input configuration file path.")
        raise ValueError

    set_log_handles(log_level, Path(log_path))
    jdata = j_loader(INPUT)

    # Complete the config jdata file
    if not INPUT == training_config_path:
        if output:
            jdata["train_options"]["output"] = str(Path(output).absolute())
            jdata["train_options"]["checkpoint_path"] = str(Path(checkpoint_path).absolute())

        if init_model:
            jdata["train_options"]["mode"] = "init_model"
            jdata["train_options"]["init_path"] = str(Path(init_model).absolute())
        elif restart:
            jdata["train_options"]["mode"] = "restart"
            jdata["train_options"]["init_path"] = str(Path(restart).absolute())
        else:
            jdata["train_options"]["mode"] = "from_scratch"

        if log_path:
            jdata["train_options"]["log_path"] = str(Path(log_path).absolute())

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
