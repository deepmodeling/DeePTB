import  torch
from dptb.utils.constants import atomic_num_dict_r
from dptb.nnsktb.formula import SKFormula
import re
from dptb.utils.index_mapping import Index_Mapings
import logging
from dptb.nnsktb.skintTypes import all_skint_types, all_onsite_intgrl_types

log = logging.getLogger(__name__)

def init_from_model_(SKNet, checkpoint_list, interpolate=False):
    """
        需要考虑的加载模式：
        1. 最普遍的，读到checkpoint里有的，SKNET中需要的onsite或者hopping，直接赋值
        2. 第一个考虑的拓展：Compound 如果checkpoint中有对应的单质hopping，这时候SKNET中需要化合物的hopping，考虑插值初始化
        3. 不同元素的网络之间的相互初始化
        
        加载中需要照顾的参数设置问题：
        1. 如果checkpoint中的skent hidden neuron个数或与SKNet不同，如何：
            a. 直接赋值给sknet
            b. 插值初始化
        2. 如果checkpoint中的skent hidden neuron个数，不同单质之间的不同，如何：
            a. 直接赋值给sknet
            b. 插值初始化
    Args:
        SKNet (_type_): _description_
        checkpoint_list (_type_): _description_
    """

    onsite_types = SKNet.onsite_types.copy()
    skint_types = SKNet.skint_types.copy()
    soc_types = SKNet.soc_types.copy()

    model_state_dict = SKNet.state_dict()

    # direct update:
    types_list = [onsite_types, skint_types, soc_types]
    layers_list = [["onsite_net.layer1", "onsite_net.layer2"], ["hopping_net.layer1", "hopping_net.layer2"], ["soc_net.layer1", "soc_net.layer2"]]

    skint_types_ckpt_list = []
    skint_layers_ckpt_list = []

    for ckpt in checkpoint_list:
        ckpt_state_dict = ckpt["model_state_dict"]
        types_list_ckpt = ckpt["model_config"]["types_list"]
        for i in range(len(types_list)):
            # update only when SKNET has this net
            if isinstance(model_state_dict.get(layers_list[i][0], None), torch.Tensor) \
                and isinstance(ckpt_state_dict.get(layers_list[i][0], None), torch.Tensor):
                types = types_list[i]
                layers = [model_state_dict[layers_list[i][0]], model_state_dict[layers_list[i][1]]]
                types_ckpt = types_list_ckpt[i]
                layers_ckpt = [ckpt_state_dict[layers_list[i][0]], ckpt_state_dict[layers_list[i][1]]]
                if layers[0] is not None:
                    layers = copy_param(types=types, layers=layers, types_ckpt=types_ckpt, layers_ckpt=layers_ckpt)

                    model_state_dict[layers_list[i][0]], model_state_dict[layers_list[i][1]] = layers[0], layers[1]
        
            if interpolate and i==1:
                skint_types_ckpt_list.append(types_ckpt)
                skint_layers_ckpt_list.append(layers_ckpt)
    
    if interpolate:
        skint_layers = interpolate_init(
            skint_types=skint_types, 
            skint_layers=[model_state_dict[layers_list[1][0]], model_state_dict[layers_list[1][1]]], 
            skint_types_ckpt_list=skint_types_ckpt_list,
            skint_layers_ckpt_list = skint_layers_ckpt_list
            )

        model_state_dict[layers_list[1][0]], model_state_dict[layers_list[1][1]] = skint_layers[0], skint_layers[1]
    
    SKNet.load_state_dict(model_state_dict)
    return SKNet

def init_from_json_(SKNet, json_model:dict):
    """
    从json 文件初始化网络参数,json文件包含了对应key的网络输出值。利用输出的值, 设定nhidden=1, 来初始化网络对应的参数。
    Args:
        SKNet (_type_): sknet model
        json_model (dict): {"onsite": onsitedict, "hopping": hopppingdict, "soc": socdict]
    """
    onsite_types = SKNet.onsite_types.copy()
    skint_types = SKNet.skint_types.copy()
    soc_types = SKNet.soc_types.copy()

    model_state_dict = SKNet.state_dict()
    
    types_list = [onsite_types, skint_types, soc_types]
    types_list_names = ['onsite_types', 'skint_types', 'soc_types']
    layers_list = [["onsite_net.layer1", "onsite_net.layer2"], ["hopping_net.layer1", "hopping_net.layer2"], ["soc_net.layer1", "soc_net.layer2"]]

    json_model_types = ["onsite", "hopping","soc"]
    #assert "onsite" in json_model.keys() and "hopping" in json_model.keys()
    
    for i in range(len(types_list)):
        # update only when SKNET using the json data.
        if isinstance(model_state_dict.get(layers_list[i][0], None), torch.Tensor) \
                and json_model_types[i] in json_model:

            types = types_list[i] # the list of net key, types. as N-N-2s-2s-0, or N-B-2s-2p-1
            typename = types_list_names[i]
            layers = [model_state_dict[layers_list[i][0]], model_state_dict[layers_list[i][1]]] 
            json_model_i = json_model[json_model_types[i]]

            if layers[0] is not None:
                layers = init_para_from_out(typename=typename, types=types, layers=layers, json_model=json_model_i)

                model_state_dict[layers_list[i][0]], model_state_dict[layers_list[i][1]] = layers[0], layers[1]

    SKNet.load_state_dict(model_state_dict)
    return SKNet

def init_para_from_out(typename, types, layers, json_model):
    """ json中保存了网络对应的输出参数的数值。现在根据输出数值，逆向对网络参数进行初始化，在不影响网络性质及使用范围的情况下，设置nhidden=1，
    从而方便从输出数值进行反向初始化网络参数。

    注，sknet的网络的格式目前设置的为 layer1: [nin，1，nhidden]; layer2: [nin, nout, nhidden].
    Args:
        types (list): the list of net key, types. as N-N-2s-2s-0, or N-B-2s-2p-1
            the keys of the network, and the aslo the name of the SKTB model parameter.
        layers (list): [layer1, layer2]
            the layers of the network. both of them are tensors.
        json_model (dict): {" N-N-2s-2s-0": [float,float,...], "N-B-2s-2p-1": [float,float,...], ...}
            the json model from input.

    """
    nhidden  = layers[0].shape[2]
    assert nhidden == 1, 'for init from json files the nhidden only support 1.'
    # layer1 shape: [nin, 1, nhidden]
    # layer2 shape: [nin, nout, nhidden]
    layers1 = layers[0]
    layers2 = layers[1]
    for i in range(len(types)):
        type = types[i]
        if type in json_model.keys():
            assert layers2.shape[1] == len(json_model[type]), 'the output of the json model is not match the net output.'
            layers2[i] = torch.reshape(json_model[type],[-1,1])
        else:
            log.warning(msg= "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
            log.warning(msg= f"The type {type} for {typename} is not in the json model.")
            log.warning(msg= "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")


    layers = [layers1, layers2]

    return layers

def copy_param(types, layers, types_ckpt, layers_ckpt):
    """_summary_

    Args:
        types (_type_): _description_
        layers (list(torch.Tensor)): [layer1, layer2] of target param where each layer is a torch.Tensor type param
        types_ckpt (_type_): _description_
        layers_ckpt (list(torch.Tensor)): [layer1, layer2] of checkpoint param where each layer is a torch.Tensor type param
    """

    nhidden = layers[0].shape[2]
    nhidden_ckpt = layers_ckpt[0].shape[2]
    assert nhidden >= nhidden_ckpt, "The parameter setting both checkpoint and current network is mismatched"
    

    layer1 = layers[0]
    layer2 = layers[1]
    for i in range(len(types)):
        type = types[i]
        if type in types_ckpt:
            index = types_ckpt.index(type)
            layer1[i][:,:nhidden_ckpt] =  layers_ckpt[0][index]
            layer1[i][:,nhidden_ckpt:] = torch.randn_like(layer1[i][:,nhidden_ckpt:]) * 1e-2
            layer2[i][:,:nhidden_ckpt] = layers_ckpt[1][index]
            layer2[i][:,nhidden_ckpt:] = torch.randn_like(layer2[i][:,nhidden_ckpt:]) * 1e-2

    
    
    layers = [layer1, layer2]
    
    return layers

def interpolate_init(skint_types, skint_layers, skint_types_ckpt_list, skint_layers_ckpt_list):

    nhidden = skint_layers[0].shape[2]
    for skint_layers_ckpt in skint_layers_ckpt_list:
        assert nhidden >= skint_layers_ckpt[0].shape[2], "The radial dependency of hoppings of checkpoint and current network is mismatched"
    
    skint_types = _remove_orbnum_(skint_types)
    skint_types_ckpt_list = [_remove_orbnum_(x) for x in skint_types_ckpt_list]

    layer1 = skint_layers[0]
    layer1_ = layer1.clone()
    layer2 = skint_layers[1]
    layer2_ = layer2.clone()
    count = [0 for _ in range(len(layer1))]

    types_ckpt_all = []
    for types in skint_types_ckpt_list:
        types_ckpt_all.extend(types)

    for i in range(len(skint_types)):
        type = skint_types[i].split("-")
        t1 = "-".join([type[0],type[0]]+type[2:])
        t2 = "-".join([type[1],type[1]]+type[2:])
        t3 = "-".join([type[0],type[0]]+[type[3], type[2], type[4]])
        t4 = "-".join([type[1],type[1]]+[type[3], type[2], type[4]])
        if type[0] == type[1] or type in types_ckpt_all:
            count[i] = None
        elif types_ckpt_all.count(t1) > 1:
            count[i] = None
            log.warning(msg="Warning! There is more than one of element {0}'s param in checkpoint, interpolation on {1} is not performed.".format(type[0], type[0]+"-"+type[2]))
        elif types_ckpt_all.count(t2) > 1:
            count[i] = None
            log.warning(msg="Warning! There is more than one of element {0}'s param in checkpoint, interpolation on {1} is not performed.".format(type[2], type[0]+"-"+type[2]))
        elif types_ckpt_all.count(t1) + types_ckpt_all.count(t3) > 1 and type[3] != type[2]:
            count[i] = None
            log.warning(msg="Warning! There is more than one of element {0}'s param in checkpoint, interpolation on {1} is not performed.".format(type[0], type[0]+"-"+type[2]))
        elif types_ckpt_all.count(t2) + types_ckpt_all.count(t4) > 1 and type[3] != type[2]:
            count[i] = None
            log.warning(msg="Warning! There is more than one of element {0}'s param in checkpoint, interpolation on {1} is not performed.".format(type[0], type[0]+"-"+type[2]))
            

    for i in range(len(skint_types)):
        type = skint_types[i].split("-") # like N-B-2s-2p-0, so corresponding key will be N-N-2s-2p-0 and B-B-2s-2p-0
        if count[i] is not None:
            t1 = "-".join([type[0],type[0]]+type[2:])
            t2 = "-".join([type[1],type[1]]+type[2:])
            t3 = "-".join([type[0],type[0]]+[type[3], type[2], type[4]])
            t4 = "-".join([type[1],type[1]]+[type[3], type[2], type[4]])
            for (skint_layers_ckpt, skint_types_ckpt) in zip(skint_layers_ckpt_list, skint_types_ckpt_list):
                nhidden_ckpt = skint_layers_ckpt[0].shape[2]
                if t1 in skint_types_ckpt:
                    index = skint_types_ckpt.index(t1)
                    layer1_[i][:,:nhidden_ckpt] = skint_layers_ckpt[0][index]
                    layer2_[i][:,:nhidden_ckpt] = skint_layers_ckpt[1][index]
                    count[i] += 1
                elif t3 in skint_types_ckpt:
                    index = skint_types_ckpt.index(t3)
                    layer1_[i][:,:nhidden_ckpt] = skint_layers_ckpt[0][index]
                    layer2_[i][:,:nhidden_ckpt] = skint_layers_ckpt[1][index]
                    count[i] += 1
                if t2 in skint_types_ckpt:
                    index = skint_types_ckpt.index(t2)
                    layer1_[i][:,:nhidden_ckpt] = skint_layers_ckpt[0][index]
                    layer2_[i][:,:nhidden_ckpt] = skint_layers_ckpt[1][index]
                    count[i] += 1
                elif t4 in skint_types_ckpt:
                    index = skint_types_ckpt.index(t4)
                    layer1_[i][:,:nhidden_ckpt] = skint_layers_ckpt[0][index]
                    layer2_[i][:,:nhidden_ckpt] = skint_layers_ckpt[1][index]
                    count[i] += 1
    
    for i in range(len(count)):
        if count[i] == 2:
            layer1[i] = layer1_[i] * 0.5
            layer2[i] = layer2_[i] * 0.5
    
    skint_layers = [layer1, layer2]

    return skint_layers

def _remove_orbnum_(types):
    for i in range(len(types)):
        temp = types[i].split("-") # "["N", "N", "2s", "2p", "0"]"
        temp[2] = "".join(re.findall(r'[A-za-z*]', temp[2]))
        temp[3] = "".join(re.findall(r'[A-za-z*]', temp[3]))
        types[i] = "-".join(temp)
    
    return types