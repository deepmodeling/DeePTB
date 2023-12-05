from dptb.nn.deeptb import DPTB, MIX
import logging
from dptb.nn.nnsk import NNSK
import torch
from dptb.utils.tools import j_must_have

log = logging.getLogger(__name__)

def build_model(run_options, model_options: dict={}, common_options: dict={}):
    """
    The build model method should composed of the following steps:
        1. process the configs from user input and the config from the checkpoint (if any).
        2. construct the model based on the configs.
        3. process the config dict for the output dict.
        run_opt = {
        "init_model": init_model,
        "restart": restart,
        "freeze": freeze,
        "log_path": log_path,
        "log_level": log_level,
        "use_correction": use_correction
    }
    """
    # this is the 
    # process the model_options
    assert not all((run_options.get("init_model"), run_options.get("restart"))), "You can only choose one of the init_model and restart options."
    if any((run_options.get("init_model"), run_options.get("restart"))):
        from_scratch = False
        checkpoint = run_options.get("init_model") or run_options.get("restart")
    else:
        from_scratch = True
        if not all((model_options, common_options)):
            logging.error("You need to provide model_options and common_options when you are initializing a model from scratch.")
            raise ValueError

    # decide whether to initialize a mixed model, or a deeptb model, or a nnsk model
    init_deeptb = False
    init_nnsk = False
    init_mixed = False

    # load the model_options and common_options from checkpoint if not provided
    if not from_scratch:
        # init model from checkpoint
        if len(model_options) == 0:
            f = torch.load(checkpoint)
            model_options = f["config"]["model_options"]
            del f

        if len(common_options) == 0:
            f = torch.load(checkpoint)
            common_options = f["config"]["common_options"]
            del f

    if all((all((model_options.get("embedding"), model_options.get("prediction"))), model_options.get("nnsk"))):
        init_mixed = True
    elif all((model_options.get("embedding"), model_options.get("prediction"))):
        init_deeptb = True
    elif model_options.get("nnsk"):
        init_nnsk = True
    else:
        log.error("Model cannot be built without either one of the terms in model_options (embedding+prediction/nnsk).")
        raise ValueError

    assert int(init_mixed) + int(init_deeptb) + int(init_nnsk) == 1, "You can only choose one of the mixed, deeptb, and nnsk options."
    # check if the model is deeptb or nnsk

    # init deeptb
    if from_scratch:
        if init_deeptb:
            model = DPTB(**model_options, **common_options)

        if init_nnsk:
            model = NNSK(**model_options["nnsk"], **common_options)

        if init_mixed:
            model = MIX(**model_options, **common_options)
            
    else:
        # load the model from the checkpoint
        if init_deeptb:
            model = DPTB.from_reference(checkpoint, **model_options, **common_options)
        if init_nnsk:
            model = NNSK.from_reference(checkpoint, **model_options["nnsk"], **common_options)
        if init_mixed:
            # mix model can be initilized with a mixed reference model or a nnsk model.
            model = MIX.from_reference(checkpoint, **model_options, **common_options)
    
    return model
