from dptb.utils.index_mapping import Index_Mapings 
from dptb.nnsktb.skintTypes import all_skint_types, all_onsite_intgrl_types
import torch

def load_paras(model_config, state_dict, proj_atom_anglr_m, onsitemode:str='none'):
    #if proj_atom_anglr_m == model_config['proj_atom_anglr_m']:
    #    return model_config, state_dict

    indmap = Index_Mapings()
    indmap.update(proj_atom_anglr_m=proj_atom_anglr_m)
    bond_index_map, bond_num_hops = indmap.Bond_Ind_Mapings()
    all_skint_types_dict, reducted_skint_types, sk_bond_ind_dict = all_skint_types(bond_index_map)
    onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num = indmap.Onsite_Ind_Mapings(onsitemode, atomtype=model_config.get("atom_type"))
    all_onsiteint_types_dcit, reducted_onsiteint_types, onsite_strain_ind_dict = all_onsite_intgrl_types(onsite_strain_index_map)

    # if onsitemode in ['uniform', 'none']: 
    #     onsite_index_map, onsite_num = indmap.Onsite_Ind_Mapings()
    # elif onsitemode == 'split':
    #     onsite_index_map, onsite_num = indmap.Onsite_Ind_Mapings_OrbSplit()
    # elif onsitemode == 'strain':
    #     onsite_index_map, onsite_num = indmap.Onsite_Ind_Mapings()
    #     onsite_strain_index_map, onsite_strain_num = indmap.OnsiteStrain_Ind_Mapings(model_config.get("atom_type"))
    #     all_onsiteint_types_dcit, reducted_onsiteint_types, onsite_strain_ind_dict = all_onsite_intgrl_types(onsite_strain_index_map)
    # else:
    #     raise ValueError('Unknown onsitemode.')
    

    proj_atom_anglr_m_ckpt = model_config['proj_atom_anglr_m']
    indmap.update(proj_atom_anglr_m=proj_atom_anglr_m_ckpt)
    bond_index_map, bond_num_hops = indmap.Bond_Ind_Mapings()
    all_skint_types_dict_ckpt, reducted_skint_types_ckpt, sk_bond_ind_dict_ckpt = all_skint_types(bond_index_map)
    onsite_strain_index_map_ckpt, onsite_strain_num_ckpt, onsite_index_map_ckpt, onsite_num_ckpt = indmap.Onsite_Ind_Mapings(model_config['onsitemode'], atomtype=model_config.get("atom_type"))
    all_onsiteint_types_dcit_ckpt, reducted_onsiteint_types_ckpt, onsite_strain_ind_dict_ckpt = all_onsite_intgrl_types(onsite_strain_index_map_ckpt)
    
    # if model_config['onsitemode'] in ['uniform', 'none']: 
    #     onsite_index_map_ckpt, onsite_num_ckpt = indmap.Onsite_Ind_Mapings()
    # elif model_config['onsitemode'] == 'split':
    #     onsite_index_map_ckpt, onsite_num_ckpt = indmap.Onsite_Ind_Mapings_OrbSplit()
    # elif model_config["onsitemode"] == 'strain':
    #     onsite_index_map_ckpt, onsite_num_ckpt = indmap.Onsite_Ind_Mapings()
    #     onsite_strain_index_map_ckpt, onsite_strain_num_ckpt = indmap.OnsiteStrain_Ind_Mapings(model_config.get("atom_type"))
    #     all_onsiteint_types_dcit_ckpt, reducted_onsiteint_types_ckpt, onsite_strain_ind_dict_ckpt = all_onsite_intgrl_types(onsite_strain_index_map_ckpt)
    # else:
    #     raise ValueError('Unknown onsitemode.')


    nhidden = model_config['sk_hop_nhidden']
    layer1 = torch.zeros([len(reducted_skint_types),nhidden])
    for i in range(len(reducted_skint_types)):
        bond_type = reducted_skint_types[i]
        if bond_type in reducted_skint_types_ckpt:
            index = reducted_skint_types_ckpt.index(bond_type)
            layer1[i] =  state_dict['bond_net.layer1'][index]
    state_dict['bond_net.layer1'] = layer1
    nhop_out = state_dict['bond_net.layer2'].shape[1]
    
    onsite_nhidden = model_config['sk_onsite_nhidden']
    
    if onsitemode == 'none':
        pass
    elif onsitemode in ['uniform', 'split']:
        for i in onsite_num.keys(): 
            layer1 = torch.zeros([1,onsite_nhidden])
            layer2 = torch.zeros([onsite_nhidden,onsite_num[i]])
            if model_config['onsitemode'] == onsitemode:
                if f'onsite_net.{i}.layer2' in  state_dict:
                    assert  onsite_nhidden == state_dict[f'onsite_net.{i}.layer2'].shape[0]
                    for orb in onsite_index_map[i].keys():
                        if (orb in onsite_index_map_ckpt[i]):
                            layer2[:,onsite_index_map[i][orb]] = \
                                state_dict[f'onsite_net.{i}.layer2'][:,onsite_index_map_ckpt[i][orb]]
                if f'onsite_net.{i}.layer1' in  state_dict:
                    layer1 = state_dict[f'onsite_net.{i}.layer1']

            state_dict[f'onsite_net.{i}.layer2'] = layer2
            state_dict[f'onsite_net.{i}.layer1'] = layer1

    elif onsitemode == 'strain':
        
        layer1 = torch.zeros([len(reducted_onsiteint_types), onsite_nhidden])
        layer2 = torch.zeros([onsite_nhidden, nhop_out])

        if model_config['onsitemode'] == onsitemode:
            if f'onsite_net.layer1' in  state_dict: 
                for i in range(len(reducted_onsiteint_types)):
                    env_type = reducted_onsiteint_types[i]
                    if env_type in reducted_onsiteint_types_ckpt:
                        index = reducted_onsiteint_types_ckpt.index(env_type)
                        layer1[i] =  state_dict['onsite_net.layer1'][index]
            if f'onsite_net.layer2' in  state_dict: 
                layer2 =  state_dict['onsite_net.layer2']

        state_dict['onsite_net.layer1'] = layer1
        state_dict['onsite_net.layer2'] = layer2

    else:
        raise ValueError('Unknown onsitemode.')
    
    model_config.update({"proj_atom_anglr_m":proj_atom_anglr_m,
                            "skint_types":reducted_skint_types,
                            "onsite_num":onsite_num})

    return model_config, state_dict
