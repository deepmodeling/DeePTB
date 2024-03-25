from dptb.nnops.trainer import Trainer
from dptb.nn.build import build_model
from dptb.data.build import build_dataset
from dptb.plugins.monitor import TrainLossMonitor, LearningRateMonitor, Validationer
from dptb.plugins.train_logger import Logger
from dptb.utils.argcheck import normalize
from dptb.plugins.saver import Saver
from typing import Dict, List, Optional, Any
from dptb.utils.tools import j_loader, setup_seed, j_must_have
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
        train_soc:bool,
        output: str,
        log_level: int,
        log_path: Optional[str],
        **kwargs
):
    run_opt = {
        "init_model": init_model,
        "restart": restart,
        "train_soc": train_soc,
        "log_path": log_path,
        "log_level": log_level
    }

    assert train_soc is False, "train_soc is not supported yet"

    '''
        -1- set up input and output directories
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
            - config.json
    '''
    # init all paths
    # if init_model, restart or init_frez, findout the input configure file

    # setup INPUT path

    if all((run_opt["init_model"], restart)):
        raise RuntimeError(
            "--init-model and --restart should not be set at the same time"
        )
    
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

    set_log_handles(log_level, Path(log_path) if log_path else None)
    # parse the config. Since if use init, config file may not equals to current
    
    jdata = j_loader(INPUT)
    jdata = normalize(jdata)
    # update basis if init_model or restart
    # update jdata
    # this is not necessary, because if we init model from checkpoint, the build_model will load the model_options from checkpoints if not provided
    # since here we want to output jdata as a config file to inform the user what model options are used, we need to update the jdata
    
    torch.set_default_dtype(getattr(torch, jdata["common_options"]["dtype"]))

    if restart or init_model:
        f = restart if restart else init_model
        f = torch.load(f)

        if jdata.get("model_options", None) is None:
            jdata["model_options"] = f["config"]["model_options"]

        # update basis
        basis = f["config"]["common_options"]["basis"]
        # nnsk
        if len(f["config"]["model_options"])==1 and f["config"]["model_options"].get("nnsk") != None:
            for asym, orb in jdata["common_options"]["basis"].items():
                assert asym in basis.keys(), f"Atom {asym} not found in model's basis"
                if orb != basis[asym]:
                    log.info(f"Initializing Orbital {orb} of Atom {asym} from {basis[asym]}")
            # we have the orbitals in jdata basis correct, now we need to make sure all atom in basis are also contained in jdata basis
            for asym, orb in basis.items():
                if asym not in jdata["common_options"]["basis"].keys():
                    jdata["common_options"]["basis"][asym] = orb # add the atomtype in the checkpoint but not in the jdata basis, because it will be used to build the orbital mapper for dataset
        else: # not nnsk
            for asym, orb in jdata["common_options"]["basis"].items():
                assert asym in basis.keys(), f"Atom {asym} not found in model's basis"
                assert orb == basis[asym], f"Orbital {orb} of Atom {asym} not consistent with the model's basis, which is only allowed in nnsk training"

            jdata["common_options"]["basis"] = basis
        
        # update model options and train_options
        if restart:
            # 
            if jdata.get("train_options", None) is not None:
                for obj in Trainer.object_keys:
                    if jdata["train_options"].get(obj) != f["config"]["train_options"].get(obj):
                        log.warning(f"{obj} in config file is not consistent with the checkpoint, using the one in checkpoint")
                        jdata["train_options"][obj] = f["config"]["train_options"][obj]
            else:
                jdata["train_options"] = f["config"]["train_options"]

            if jdata.get("model_options", None) is None or jdata["model_options"] != f["config"]["model_options"]:
                log.warning("model_options in config file is not consistent with the checkpoint, using the one in checkpoint")
                jdata["model_options"] = f["config"]["model_options"] # restart does not allow to change model options
        else:
            # init model mode, allow model_options change
            if jdata.get("train_options", None) is None:
                jdata["train_options"] = f["config"]["train_options"]
            if jdata.get("model_options") is None:
                jdata["model_options"] = f["config"]["model_options"]
        del f
    else:
        j_must_have(jdata, "model_options")
        j_must_have(jdata, "train_options")


    # setup seed
    setup_seed(seed=jdata["common_options"]["seed"])

    # with open(os.path.join(output, "train_config.json"), "w") as fp:
    #     json.dump(jdata, fp, indent=4)

    # build dataset
    train_datasets = build_dataset(set_options=jdata["data_options"]["train"], common_options=jdata["common_options"])
    if jdata["data_options"].get("validation"):
        validation_datasets = build_dataset(set_options=jdata["data_options"]["validation"], common_options=jdata["common_options"])
    else:
        validation_datasets = None
    if jdata["data_options"].get("reference"):
        reference_datasets = build_dataset(set_options=jdata["data_options"]["reference"], common_options=jdata["common_options"])
    else:
        reference_datasets = None

    if restart:
        trainer = Trainer.restart(
            train_options=jdata["train_options"],
            common_options=jdata["common_options"],
            checkpoint=restart,
            train_datasets=train_datasets,
            reference_datasets=reference_datasets,
            validation_datasets=validation_datasets,
        )
    else:
        # include the init model and from scratch
        # build model will handle the init model cases where the model options provided is not equals to the ones in checkpoint.
        model = build_model(run_options=run_opt, model_options=jdata["model_options"], common_options=jdata["common_options"], statistics=train_datasets.E3statistics())
        trainer = Trainer(
            train_options=jdata["train_options"],
            common_options=jdata["common_options"],
            model = model,
            train_datasets=train_datasets,
            validation_datasets=validation_datasets,
            reference_datasets=reference_datasets,
        )
    
    # register the plugin in trainer, to tract training info
    log_field = ["train_loss", "lr"]
    if validation_datasets:
        trainer.register_plugin(Validationer())
        log_field.append("validation_loss")
    trainer.register_plugin(TrainLossMonitor())
    trainer.register_plugin(LearningRateMonitor())
    trainer.register_plugin(Logger(log_field, 
        interval=[(jdata["train_options"]["display_freq"], 'iteration'), (1, 'epoch')]))
    
    for q in trainer.plugin_queues.values():
        heapq.heapify(q)

    if output:
        # output training configurations:
        with open(os.path.join(output, "train_config.json"), "w") as fp:
            json.dump(jdata, fp, indent=4)

        trainer.register_plugin(Saver(
            #interval=[(jdata["train_options"].get("save_freq"), 'epoch'), (1, 'iteration')] if jdata["train_options"].get(
            #    "save_freq") else None))
            interval=[(jdata["train_options"].get("save_freq"), 'iteration'),  (1, 'epoch')] if jdata["train_options"].get(
                "save_freq") else None), checkpoint_path=checkpoint_path)
        # add a plugin to save the training parameters of the model, with model_output as given path

    start_time = time.time()

    trainer.run(trainer.train_options["num_epoch"])

    end_time = time.time()
    log.info("finished training")
    log.info(f"wall time: {(end_time - start_time):.3f} s")


