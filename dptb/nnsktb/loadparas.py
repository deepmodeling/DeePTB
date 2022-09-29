from dptb.utils.index_mapping import Index_Mapings 
from dptb.nnsktb.skintTypes import all_skint_types
import torch

def load_paras(model_config, state_dict, proj_atom_anglr_m):
    if proj_atom_anglr_m == model_config['proj_atom_anglr_m']:
        return model_config, state_dict

    indmap = Index_Mapings()
    indmap.update(proj_atom_anglr_m=proj_atom_anglr_m)
    bond_index_map, bond_num_hops = indmap.Bond_Ind_Mapings()
    all_skint_types_dict, reducted_skint_types, sk_bond_ind_dict = all_skint_types(bond_index_map)
    if model_config['onsitemode'] == 'uniform': 
        onsite_index_map, onsite_num = indmap.Onsite_Ind_Mapings()
    elif model_config['onsitemode'] == 'split':
        onsite_index_map, onsite_num = indmap.Onsite_Ind_Mapings_OrbSplit()
    else:
        raise ValueError('Unknown onsitemode.')
    

    proj_atom_anglr_m_ckpt = model_config['proj_atom_anglr_m']
    indmap.update(proj_atom_anglr_m=proj_atom_anglr_m_ckpt)
    bond_index_map, bond_num_hops = indmap.Bond_Ind_Mapings()
    all_skint_types_dict_ckpt, reducted_skint_types_ckpt, sk_bond_ind_dict_ckpt = all_skint_types(bond_index_map)
    if model_config['onsitemode'] == 'uniform': 
        onsite_index_map_ckpt, onsite_num_ckpt = indmap.Onsite_Ind_Mapings()
    elif model_config['onsitemode'] == 'split':
        onsite_index_map_ckpt, onsite_num_ckpt = indmap.Onsite_Ind_Mapings_OrbSplit()
    else:
        raise ValueError('Unknown onsitemode.')

    if model_config.get("onsite_strain"):
        indmap.update(proj_atom_anglr_m=proj_atom_anglr_m)
        onsite_strain_index_map, onsite_strain_num = indmap.OnsiteStrain_Ind_Mapings(model_config.get("atom_type"))
        all_onsiteint_types_dcit, reducted_onsiteint_types, onsite_strain_ind_dict = all_skint_types(onsite_strain_index_map)

        indmap.update(proj_atom_anglr_m=proj_atom_anglr_m_ckpt)
        onsite_strain_index_map_ckpt, onsite_strain_num_ckpt = indmap.OnsiteStrain_Ind_Mapings(model_config.get("atom_type"))
        all_onsiteint_types_dcit_ckpt, reducted_onsiteint_types_ckpt, onsite_strain_ind_dict_ckpt = all_skint_types(onsite_strain_index_map_ckpt)

        nhidden = model_config['sk_hop_nhidden']
        layer1 = torch.zeros([len(reducted_onsiteint_types), nhidden])
        for i in range(len(reducted_onsiteint_types)):
            env_type = reducted_onsiteint_types[i]
            if env_type in reducted_onsiteint_types_ckpt:
                index = reducted_skint_types_ckpt.index(env_type)
                layer1[i] =  state_dict['onsite_strain_net.layer1'][index]
        state_dict['onsite_strain_net.layer1'] = layer1



    #paras['bond_net.layer1']
    nhidden = model_config['sk_hop_nhidden']
    layer1 = torch.zeros([len(reducted_skint_types),nhidden])
    for i in range(len(reducted_skint_types)):
        bond_type = reducted_skint_types[i]
        if bond_type in reducted_skint_types_ckpt:
            index = reducted_skint_types_ckpt.index(bond_type)
            layer1[i] =  state_dict['bond_net.layer1'][index]
    state_dict['bond_net.layer1'] = layer1

    onsite_nhidden = model_config['sk_onsite_nhidden']
    for i in onsite_num.keys(): 
        if f'onsite_net.{i}.layer2' in  state_dict:
            assert  onsite_nhidden == state_dict[f'onsite_net.{i}.layer2'].shape[0]
            layer2 = torch.zeros([onsite_nhidden,onsite_num[i]])
            for orb in onsite_index_map[i].keys():
                if (orb in onsite_index_map_ckpt[i]):
                    layer2[:,onsite_index_map[i][orb]] = \
                        state_dict[f'onsite_net.{i}.layer2'][:,onsite_index_map_ckpt[i][orb]]
        else:
            layer2 = torch.zeros([onsite_nhidden,onsite_num[i]])
        state_dict[f'onsite_net.{i}.layer2'] = layer2
    
    model_config.update({"proj_atom_anglr_m":proj_atom_anglr_m,
                        "skint_types":reducted_skint_types,
                        "onsite_num":onsite_num})
    
    return model_config, state_dict
