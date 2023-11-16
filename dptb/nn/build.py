from dptb.nn.deeptb import DPTB
from dptb.nn.nnsk import NNSK
from dptb.utils.tools import j_must_have

def build_model(run_options, model_options, common_options):
    """
    The build model method should composed of the following steps:
        1. process the configs from user input and the config from the checkpoint (if any).
        2. construct the model based on the configs.
        3. process the config dict for the output dict.
    """

    # this is the 
    # process the model_options
    
    model = None

    init_deeptb = False
    init_nnsk = False
    # check if the model is deeptb or nnsk
    if len(model_options.get("embedding")) != 0 and len(model_options.get("prediction")) != 0:
        init_deeptb = True
    if len(model_options.get("nnsk")) != 0:
        init_nnsk = True

    # init deeptb
    if init_deeptb:
        deeptb_model = DPTB(**model_options, **common_options)


    # init nnsk
    if init_nnsk:
        nnsk_options = j_must_have
        nnsk_model = NNSK(**nnsk_options, **common_options)

    
    return model