# this file is to check the input configuration file to run specific commands.

import os
import sys
import torch
from dptb.utils.tools import j_loader, j_must_have
import logging
from typing import Dict, List, Optional, Any
from dptb.utils.argcheck import normalize


log = logging.getLogger(__name__)
#TODO: 对于loss 和 data option 的检查还没有完整

def check_config_train(
        INPUT,
        init_model: Optional[str],
        restart: Optional[str],
        **kwargs):
    
    if all((init_model, restart)):
        raise RuntimeError("--init-model and --restart should not be set at the same time")
    
    jdata = j_loader(INPUT)
    jdata = normalize(jdata)

    if not (restart or init_model):
        j_must_have(jdata, "model_options")
        j_must_have(jdata, "train_options")

    assert j_must_have(jdata["data_options"], "train"), "train data set in data_options is not provided in the input configuration file."
    train_data_config = jdata["data_options"]["train"]

    if train_data_config.get("get_eigenvalues") and not train_data_config.get("get_Hamiltonian"):
        assert jdata['train_options']['loss_options']['train'].get("method") in ["eigvals"]

    if train_data_config.get("get_Hamiltonian") and not train_data_config.get("get_eigenvalues"):
        assert jdata['train_options']['loss_options']['train'].get("method").startswith("hamil")

    # if train_data_config.get("get_Hamiltonian") and train_data_config.get("get_eigenvalues"):
    #     raise RuntimeError("The train data set should not have both get_Hamiltonian and get_eigenvalues set to True.")

    #if jdata["data_options"].get("validation"):
    
    
    if not (restart or init_model):
        model_options = jdata["model_options"]
        if  model_options.get("nnsk"):
            if all((model_options.get("embedding"), model_options.get("prediction"))):
                init_mixed = True
                if not model_options['prediction']['method'] == 'sktb':
                    log.error("The prediction method must be sktb for mix mode.")
                    raise ValueError("The prediction method must be sktb for mix mode.")
                
                if not model_options['embedding']['method'] in ['se2']:
                    log.error("The embedding method must be se2 for mix mode.")
                    raise ValueError("The embedding method must be se2 for mix mode.")
        
            elif not any((model_options.get("embedding"), model_options.get("prediction"))):
                init_nnsk = True
            else:
                log.error("Model_options are not set correctly! \n" + 
                          "You can only choose one of the mixed, deeptb, and nnsk modes.\n" + 
                          " -  `mixed`, set all the `nnsk` `embedding` and `prediction` options.\n" +
                          " -  `deeptb`, set `embedding` and `prediction` options and no `nnsk`.\n" +
                          " -  `nnsk`, set only `nnsk` options.")
                raise ValueError("Model_options are not set correctly!")
        else:
            if all((model_options.get("embedding"), model_options.get("prediction"))):
                init_deeptb = True
                if model_options["prediction"]['method'] == 'sktb':
                    log.warning("The prediction method is sktb, but the nnsk option is not set. this is highly not recommand.\n"+
                                "We recommand to train nnsk then train mix model for sktb. \n"+
                                "Please make sure you know what you are doing!")
                    if not model_options['embedding']['method'] in ['se2']:
                        log.error("The embedding method must be se2 for sktb prediction in deeptb mode.")
                        raise ValueError("The embedding method must be se2 for sktb prediction in deeptb mode.")
                if model_options["prediction"]['method'] == 'e3tb':
                    # 对于E3 statistics 一定会设置的吗？
                    # if statistics is None:
                    #    log.error("The statistics must be provided for e3tb prediction method.")
                    #     raise ValueError("The statistics must be provided for e3tb prediction method.")
                    if  model_options['embedding']['method'] in ['se2']:
                        log.error("The embedding method can not be se2 for e3tb prediction in deeptb mode.")
                        raise ValueError("The embedding method can not be se2 for e3tb prediction in deeptb mode.")
            else:
                log.error("Model_options are not set correctly! \n" + 
                          "You can only choose one of the mixed, deeptb, and nnsk modes.\n" + 
                          " -  `mixed`, set all the `nnsk` `embedding` and `prediction` options.\n" +
                          " -  `deeptb`, set `embedding` and `prediction` options and no `nnsk`.\n" +
                          " -  `nnsk`, set only `nnsk` options.")
                raise ValueError("Model_options are not set correctly!")
        
        #if jdata["data_options"].get("reference"):
        #    log.info("reference set is provided. Then the loss options should have set the reference loss options.")