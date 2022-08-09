from dptb.nnops.model_train import Trainer
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
        output: str,
        log_level: int,
        log_path: Optional[str],
        **kwargs
):
    # init all paths
    if all((init_model, restart)):
        raise RuntimeError(
            "--init-model and --restart should not be set at the same time"
        )
    if output:
        Path(output).mkdir(exist_ok=True, parents=True)
        Path(output).parent.mkdir(exist_ok=True, parents=True)
        checkpoint_path = os.path.join(str(output), "checkpoint")
        Path(checkpoint_path).mkdir(exist_ok=True, parents=True)
    if log_path:
        log_path = log_path
    elif output:
        log_path = os.path.join(str(output), "log/log.txt")

    set_log_handles(log_level, Path(log_path))
    jdata = j_loader(INPUT)

    # record training configuration
    if output:
        jdata["train_options"]["output"] = str(Path(output).absolute())
        jdata["train_options"]["checkpoint_path"] = str(Path(checkpoint_path).absolute())

    if init_model:
        jdata["train_options"]["mode"] = "from_model"
        jdata["train_options"]["init_path"] = str(Path(init_model).absolute())
    else:
        jdata["train_options"]["mode"] = "from_scratch"

    if log_path:
        jdata["train_options"]["log_path"] = str(Path(log_path).absolute())


    trainer = Trainer(**jdata)

    # register the plugin in trainer, to tract training info
    trainer.register_plugin(Validationer())
    trainer.register_plugin(TrainLossMonitor())
    trainer.register_plugin(LearningRateMonitor())
    trainer.register_plugin(Logger(["train_loss","validation_loss","lr"]))

    if output:
        # output training configurations:
        with open(os.path.join(output, "train_config.json"), "w") as fp:
            json.dump(jdata, fp, indent=4)

        trainer.register_plugin(Saver(interval=[(jdata["train_options"].get("save_epoch"), 'epoch')] if jdata["train_options"].get("save_epoch") else None))
        # add a plugin to save the training parameters of the model, with model_output as given path






    start_time = time.time()

    trainer.run(trainer.num_epoch)

    end_time = time.time()
    log.info("finished training")
    log.info(f"wall time: {(end_time - start_time):.3f} s")
