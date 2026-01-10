import heapq
import logging
import torch
import json
import os
import time
from pathlib import Path
from dptb.nn.build import build_model
from dptb.data.build import build_dataset
from typing import Optional
from dptb.utils.loggers import set_log_handles
from dptb.utils.tools import j_loader, setup_seed
from dptb.nnops.tester import Tester
from dptb.utils.argcheck import normalize_test
from dptb.plugins.monitor import TestLossMonitor
from dptb.plugins.train_logger import Logger

__all__ = ["test"]

log = logging.getLogger(__name__)

def _test(
        INPUT: str,
        init_model: str,
        output: str,
        log_level: int,
        log_path: Optional[str],
        use_correction: Optional[str],
        **kwargs
):
    # TODO: permit commandline init_model and config file init.
    run_opt = {
        "init_model": init_model,
        "log_path": log_path,
        "log_level": log_level,
        "use_correction": use_correction,
        "freeze":True,
        "train_soc":False
    }
    
    # setup output path
    if output:
        Path(output).parent.mkdir(exist_ok=True, parents=True)
        Path(output).mkdir(exist_ok=True, parents=True)
        results_path = os.path.join(str(output), "results")
        Path(results_path).mkdir(exist_ok=True, parents=True)
        if not log_path:
            log_path = os.path.join(str(output), "log/log.txt")
        Path(log_path).parent.mkdir(exist_ok=True, parents=True)

        run_opt.update({
                        "output": str(Path(output).absolute()),
                        "results_path": str(Path(results_path).absolute()),
                        "log_path": str(Path(log_path).absolute())
                        })
    
    jdata = j_loader(INPUT)
    jdata = normalize_test(jdata)
    # setup seed
    setup_seed(seed=jdata["common_options"]["seed"])

    f = torch.load(init_model, weights_only=False)
    # update basis
    basis = f["config"]["common_options"]["basis"]
    for asym, orb in jdata["common_options"]["basis"].items():
        assert asym in basis.keys(), f"Atom {asym} not found in model's basis"
        assert orb == basis[asym], f"Orbital {orb} of Atom {asym} not consistent with the model's basis"

    jdata["common_options"]["basis"] = basis # use the old basis, because it will be used to build the orbital mapper for dataset

    set_log_handles(log_level, Path(log_path) if log_path else None)

    f = torch.load(run_opt["init_model"], weights_only=False)
    jdata["model_options"] = f["config"]["model_options"]
    del f
    
    test_datasets = build_dataset(**jdata["data_options"]["test"], **jdata["common_options"])
    model = build_model(run_opt["init_model"], model_options=jdata["model_options"], common_options=jdata["common_options"])
    model.eval()
    tester = Tester(
        test_options=jdata["test_options"],
        common_options=jdata["common_options"],
        model = model,
        test_datasets=test_datasets,
    )

    # register the plugin in tester, to tract training info
    tester.register_plugin(TestLossMonitor())
    tester.register_plugin(Logger(["test_loss"], 
        interval=[(1, 'iteration'), (1, 'epoch')]))
    
    for q in tester.plugin_queues.values():
        heapq.heapify(q)
    
    tester.build()

    if output:
        # output training configurations:
        with open(os.path.join(output, "test_config.json"), "w") as fp:
            json.dump(jdata, fp, indent=4)

    start_time = time.time()

    tester.run()

    end_time = time.time()
    log.info("finished testing")
    log.info(f"wall time: {(end_time - start_time):.3f} s")