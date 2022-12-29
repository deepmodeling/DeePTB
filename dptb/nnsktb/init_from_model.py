import  torch
from dptb.utils.constants import atomic_num_dict_r
from dptb.nnsktb.formula import SKFormula
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

    onsite_types = SKNet.onsite_types
    skint_types = SKNet.skint_types
    soc_types = SKNet.soc_types

    model_state_dict = SKNet.state_dict()

    # direct update:
    types_list = [onsite_types, skint_types, soc_types]
    layers_list = [["onsite_net.layer1", "onsite_net.layer2"], ["bond_net.layer1", "bond_net.layer2"], ["soc_net.layer1", "soc_net.layer2"]]

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

    layer1 = skint_layers[0]
    layer1_ = layer1.clone()
    layer2 = skint_layers[1]
    layer2_ = layer2.clone()
    count = [0 for _ in range(len(layer1))]

    types_ckpt_all = []
    for types in skint_types_ckpt_list:
        types_ckpt_all.extend(types)

    for i in range(len(skint_types)):
        type = skint_types[i]
        t1 = type[0]+"-"+type[0]+type[3:]
        t2 = type[2]+"-"+type[2]+type[3:]
        if type[0] == type[2] or type in types_ckpt_all:
            count[i] = None
        elif types_ckpt_all.count(t1) > 1:
            count[i] = None
            log.warning(msg="Warning! There is more than one of element {0}'s param in checkpoint, interpolation on {1} is not performed.".format(type[0], type[0]+"-"+type[2]))
        elif types_ckpt_all.count(t2) > 1:
            count[i] = None
            log.warning(msg="Warning! There is more than one of element {0}'s param in checkpoint, interpolation on {1} is not performed.".format(type[2], type[0]+"-"+type[2]))
            

    for i in range(len(skint_types)):
        type = skint_types[i] # like N-B-2s-2p-0, so corresponding key will be N-N-2s-2p-0 and B-B-2s-2p-0
        iatom = type[0]
        jatom = type[2]
        if count[i] is not None:
            t1 = iatom+"-"+iatom+type[3:]
            t2 = jatom+"-"+jatom+type[3:]
            for (skint_layers_ckpt, skint_types_ckpt) in zip(skint_layers_ckpt_list, skint_types_ckpt_list):
                nhidden_ckpt = skint_layers_ckpt[0].shape[2]
                if t1 in skint_types_ckpt:
                    index = skint_types_ckpt.index(t1)
                    layer1_[i][:,:nhidden_ckpt] = skint_layers_ckpt[0][index]
                    layer2_[i][:,:nhidden_ckpt] = skint_layers_ckpt[1][index]
                    count[i] += 1
                if t2 in skint_types_ckpt:
                    index = skint_types_ckpt.index(t2)
                    layer1_[i][:,:nhidden_ckpt] = skint_layers_ckpt[0][index]
                    layer2_[i][:,:nhidden_ckpt] = skint_layers_ckpt[1][index]
                    count[i] += 1
    
    for i in range(len(count)):
        if i == 2:
            layer1[i] = layer1_[i] * 0.5
            layer2[i] = layer2_[i] * 0.5
    
    skint_layers = [layer1, layer2]
    print(count)

    return skint_layers
