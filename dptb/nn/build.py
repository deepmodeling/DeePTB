from dptb.nn.deeptb import DPTB, MIX
import logging
from dptb.nn.nnsk import NNSK
import torch
from dptb.utils.tools import j_must_have

log = logging.getLogger(__name__)

def build_model(run_options, model_options: dict={}, common_options: dict={}, statistics: dict=None):
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
            raise ValueError("You need to provide model_options and common_options when you are initializing a model from scratch.")

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
    
    
    assert int(init_mixed) + int(init_deeptb) + int(init_nnsk) == 1, "You can only choose one of the mixed, deeptb, and nnsk options."
    # check if the model is deeptb or nnsk

    # init deeptb
    if from_scratch:
        if init_deeptb:
            model = DPTB(**model_options, **common_options)

            # do initialization from statistics if DPTB is e3tb and statistics is provided
            if model.method == "e3tb" and statistics is not None:
                scalar_mask = torch.BoolTensor([ir.dim==1 for ir in model.idp.orbpair_irreps])
                node_shifts = statistics["node"]["scalar_ave"]
                node_scales = statistics["node"]["norm_ave"]
                node_scales[:,scalar_mask] = statistics["node"]["scalar_std"]

                edge_shifts = statistics["edge"]["scalar_ave"]
                edge_scales = statistics["edge"]["norm_ave"]
                edge_scales[:,scalar_mask] = statistics["edge"]["scalar_std"]
                model.node_prediction_h.set_scale_shift(scales=node_scales, shifts=node_shifts)
                model.edge_prediction_h.set_scale_shift(scales=edge_scales, shifts=edge_shifts)

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
    
    if model.model_options != model_options:
        # log.error("The model options are not consistent with the checkpoint, using the one in the checkpoint.")
        raise ValueError("The model options are not consistent with the checkpoint, using the one in the checkpoint.")
    return model
