from dptb.nnops.train_dptb import DPTBTrainer
from dptb.nnops.train_nnsk import NNSKTrainer
from dptb.plugins.monitor import TrainLossMonitor, LearningRateMonitor, Validationer
from dptb.plugins.init_nnsk import InitSKModel
from dptb.plugins.init_dptb import InitDPTBModel
from dptb.plugins.init_data import InitData
from dptb.plugins.train_logger import Logger
from dptb.utils.argcheck import normalize
from dptb.plugins.plugins import Saver
from typing import Dict, List, Optional, Any
from dptb.utils.tools import j_loader, setup_seed
from dptb.utils.constants import dtype_dict
from dptb.utils.loggers import set_log_handles
import heapq
import logging
import torch
import random
import numpy as np
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
        freeze:bool,
        train_soc:bool,
        output: str,
        log_level: int,
        log_path: Optional[str],
        train_sk: bool,
        use_correction: Optional[str],
        **kwargs
):
    run_opt = {
        "init_model": init_model,
        "restart": restart,
        "freeze": freeze,
        "train_soc": train_soc,
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
    # init all paths
    # if init_model, restart or init_frez, findout the input configure file
    
    if all((use_correction, train_sk)):
        raise RuntimeError(
            "--use-correction and --train_sk should not be set at the same time"
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
            log.info(msg="Haven't assign a initializing mode, training from scratch as default.")
            mode = "from_scratch"
            skconfig_path = INPUT
        else:
            log.error("ValueError: Missing Input configuration file path.")
            raise ValueError

        # switch the init model mode from command line to config file
        jdata = j_loader(INPUT)
        jdata = normalize(jdata)

        # check if init_model in commandline and input json are in conflict.

        if all((jdata["init_model"]["path"], run_opt["init_model"])) or \
        all((jdata["init_model"]["path"], run_opt["restart"])):
            raise RuntimeError(
                "init-model in config and command line is in conflict, turn off one of then to avoid this error !"
            )
        
        if jdata["init_model"]["path"] is not None:
            assert mode == "from_scratch"
            run_opt["init_model"] = jdata["init_model"]
            mode = "init_model"
            if isinstance(run_opt["init_model"]["path"], str):
                skconfig_path = os.path.join(str(Path(run_opt["init_model"]["path"]).parent.absolute()), "config_nnsktb.json")
            else: # list
                skconfig_path = [os.path.join(str(Path(path).parent.absolute()), "config_nnsktb.json") for path in run_opt["init_model"]["path"]]
        elif run_opt["init_model"] is not None:
            # format run_opt's init model to the format of jdata
            assert mode == "init_model"
            path = run_opt["init_model"]
            run_opt["init_model"] = jdata["init_model"]
            run_opt["init_model"]["path"] = path

        # handling exceptions when init_model path in config file is [] and [single file]
        if mode == "init_model":
            if isinstance(run_opt["init_model"]["path"], list):
                if len(run_opt["init_model"]["path"])==0:
                    raise RuntimeError("Error! list mode init_model in config file cannot be empty!")

    else:
        if init_model:
            dptbconfig_path = os.path.join(str(Path(init_model).parent.absolute()), "config_dptbtb.json")
            mode = "init_model"
        elif restart:
            dptbconfig_path = os.path.join(str(Path(restart).parent.absolute()), "config_dptbtb.json")
            mode = "restart"
        elif INPUT is not None:
            log.info(msg="Haven't assign a initializing mode, training from scratch as default.")
            dptbconfig_path = INPUT
            mode = "from_scratch"
        else:
            log.error("ValueError: Missing Input configuration file path.")
            raise ValueError

        if use_correction:
            skconfig_path = os.path.join(str(Path(use_correction).parent.absolute()), "config_nnsktb.json")
            # skcheckpoint_path = str(Path(str(input(f"Enter skcheckpoint_path (default ./checkpoint/best_nnsk.pth): \n"))).absolute())
        else:
            skconfig_path = None

        # parse INPUT file
        jdata = j_loader(INPUT)
        jdata = normalize(jdata)

        if all((jdata["init_model"]["path"], run_opt["init_model"])) or \
        all((jdata["init_model"]["path"], run_opt["restart"])):
            raise RuntimeError(
                "init-model in config and command line is in conflict, turn off one of then to avoid this error !"
            )
        
        if jdata["init_model"]["path"] is not None:
            assert mode == "from_scratch"
            log.info(msg="Init model is read from config rile.")
            run_opt["init_model"] = jdata["init_model"]
            mode = "init_model"
            if isinstance(run_opt["init_model"]["path"], str):
                dptbconfig_path = os.path.join(str(Path(run_opt["init_model"]["path"]).parent.absolute()), "config_dptb.json")
            else: # list
                raise RuntimeError(
                "loading lists of checkpoints is only supported in init_nnsk!"
            )
        elif run_opt["init_model"] is not None:
            assert mode == "init_model"
            path = run_opt["init_model"]
            run_opt["init_model"] = jdata["init_model"]
            run_opt["init_model"]["path"] = path

        if mode == "init_model":
            if isinstance(run_opt["init_model"]["path"], list):
                if len(run_opt["init_model"]["path"])==0:
                    log.error(msg="Error, no checkpoint supplied!")
                    raise RuntimeError
                elif len(run_opt["init_model"]["path"])>1:
                    log.error(msg="Error! list mode init_model in config only support single file in DPTB!")
                    raise RuntimeError

    if all((run_opt["init_model"], restart)):
        raise RuntimeError(
            "--init-model and --restart should not be set at the same time"
        )
    
    if mode == "init_model":
        if isinstance(run_opt["init_model"]["path"], list):
            if len(run_opt["init_model"]["path"]) == 1:
                run_opt["init_model"]["path"] = run_opt["init_model"]["path"][0]
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
                "skconfig_path": skconfig_path
            })
        run_opt.update({
            "dptbconfig_path": dptbconfig_path
        })

    set_log_handles(log_level, Path(log_path) if log_path else None)
    # parse the config. Since if use init, config file may not equals to current
    
    
    # setup seed
    setup_seed(seed=jdata["train_options"]["seed"])

    # with open(os.path.join(output, "train_config.json"), "w") as fp:
    #     json.dump(jdata, fp, indent=4)
    
    str_dtype = jdata["common_options"]["dtype"]
    jdata["common_options"]["dtype"] = dtype_dict[jdata["common_options"]["dtype"]]
    if train_sk:
        trainer = NNSKTrainer(run_opt, jdata)
        trainer.register_plugin(InitSKModel())
    else:
        trainer = DPTBTrainer(run_opt, jdata)
        trainer.register_plugin(InitDPTBModel())
    
    
    
    # register the plugin in trainer, to tract training info
    trainer.register_plugin(InitData())
    trainer.register_plugin(Validationer())
    trainer.register_plugin(TrainLossMonitor())
    trainer.register_plugin(LearningRateMonitor())
    trainer.register_plugin(Logger(["train_loss", "validation_loss", "lr"], 
        interval=[(jdata["train_options"]["display_freq"], 'iteration'), (1, 'epoch')]))
    
    for q in trainer.plugin_queues.values():
        heapq.heapify(q)
    
    trainer.build()


    if output:
        # output training configurations:
        with open(os.path.join(output, "train_config.json"), "w") as fp:
            jdata["common_options"]["dtype"] = str_dtype
            json.dump(jdata, fp, indent=4)

        trainer.register_plugin(Saver(
            #interval=[(jdata["train_options"].get("save_freq"), 'epoch'), (1, 'iteration')] if jdata["train_options"].get(
            #    "save_freq") else None))
            interval=[(jdata["train_options"].get("save_freq"), 'iteration'),  (1, 'epoch')] if jdata["train_options"].get(
                "save_freq") else None))
        # add a plugin to save the training parameters of the model, with model_output as given path

    start_time = time.time()

    trainer.run(trainer.num_epoch)

    end_time = time.time()
    log.info("finished training")
    log.info(f"wall time: {(end_time - start_time):.3f} s")


