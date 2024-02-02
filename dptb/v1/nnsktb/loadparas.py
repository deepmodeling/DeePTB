from dptb.utils.index_mapping import Index_Mapings 
from dptb.nnsktb.skintTypes import all_skint_types, all_onsite_intgrl_types, all_onsite_ene_types
import torch

eps = 1e-5

def load_paras(model_config, state_dict, proj_atom_anglr_m, onsitemode:str='none', soc=False):

    indmap = Index_Mapings()
    indmap.update(proj_atom_anglr_m=proj_atom_anglr_m)
    bond_index_map, bond_num_hops = indmap.Bond_Ind_Mapings()
    all_skint_types_dict, reducted_skint_types, sk_bond_ind_dict = all_skint_types(bond_index_map)
    onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num = indmap.Onsite_Ind_Mapings(onsitemode, atomtype=model_config["atomtype"])
    _, reduced_onsiteE_types, onsiteE_ind_dict = all_onsite_ene_types(onsite_index_map)
    if onsitemode == 'strain':
        all_onsiteint_types_dcit, reducted_onsiteint_types, onsite_strain_ind_dict = all_onsite_intgrl_types(onsite_strain_index_map)


    proj_atom_anglr_m_ckpt = model_config['proj_atom_anglr_m']
    indmap.update(proj_atom_anglr_m=proj_atom_anglr_m_ckpt)
    bond_index_map, bond_num_hops = indmap.Bond_Ind_Mapings()
    all_skint_types_dict_ckpt, reducted_skint_types_ckpt, sk_bond_ind_dict_ckpt = all_skint_types(bond_index_map)
    onsite_strain_index_map_ckpt, onsite_strain_num_ckpt, onsite_index_map_ckpt, onsite_num_ckpt = indmap.Onsite_Ind_Mapings(model_config['onsitemode'], atomtype=model_config["atomtype"])
    _, reduced_onsiteE_types_ckpt, onsiteE_ind_dict_ckpt = all_onsite_ene_types(onsite_index_map_ckpt)
    if model_config["onsitemode"] == "strain":
        all_onsiteint_types_dcit_ckpt, reducted_onsiteint_types_ckpt, onsite_strain_ind_dict_ckpt = all_onsite_intgrl_types(onsite_strain_index_map_ckpt)

    nhidden = model_config['sknetwork']['sk_hop_nhidden']
    layer1 = torch.randn([len(reducted_skint_types), 1, nhidden]) * eps
    for i in range(len(reducted_skint_types)):
        bond_type = reducted_skint_types[i]
        if bond_type in reducted_skint_types_ckpt:
            index = reducted_skint_types_ckpt.index(bond_type)
            layer1[i] =  state_dict['bond_net.layer1'][index]
    state_dict['bond_net.layer1'] = layer1
    nhop_out = state_dict['bond_net.layer2'].shape[1]
    
    onsite_nhidden = model_config['sknetwork']['sk_onsite_nhidden']
    if soc:
        soc_nhidden = model_config['sknetwork'].get('sk_soc_nhidden', onsite_nhidden)

    
    if onsitemode == 'none':
        pass
    elif onsitemode in ['uniform', 'split']:
        layer1 = torch.randn([len(reduced_onsiteE_types),onsite_nhidden]) * eps
        layer2 = torch.randn([onsite_nhidden, 1]) * eps
        if model_config['onsitemode'] == onsitemode:
            if f'onsite_net.layer1' in  state_dict:
                assert  onsite_nhidden == state_dict[f'onsite_net.layer1'].shape[1]
                for i in range(len(reduced_onsiteE_types)):
                    onsite_type = reduced_onsiteE_types[i]
                    if onsite_type in reduced_onsiteE_types_ckpt:
                        index = reduced_onsiteE_types_ckpt.index(onsite_type)
                        layer1[i] =  state_dict['onsite_net.layer1'][index]
    
            if f'onsite_net.layer2' in state_dict:
                layer2 = state_dict[f'onsite_net.layer2']

        state_dict[f'onsite_net.layer2'] = layer2
        state_dict[f'onsite_net.layer1'] = layer1

    elif onsitemode == 'strain':
        
        layer1 = torch.randn([len(reducted_onsiteint_types), 1, onsite_nhidden]) * eps
        layer2 = torch.randn([len(reducted_onsiteint_types), nhop_out, onsite_nhidden]) * eps
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
    
    if soc:
        layer1 = torch.randn([len(reduced_onsiteE_types), soc_nhidden]) * eps
        layer2 = torch.randn([soc_nhidden, 1]) * eps
        
        if f'soc_net.layer1' in  state_dict:
            assert  onsite_nhidden == state_dict[f'soc_net.layer1'].shape[1]
            for i in range(len(reduced_onsiteE_types)):
                soc_type = reduced_onsiteE_types[i]
                if soc_type in reduced_onsiteE_types_ckpt:
                    index = reduced_onsiteE_types_ckpt.index(soc_type)
                    layer1[i] =  state_dict['soc_net.layer1'][index]

        if f'soc_net.layer2' in state_dict:
            layer2 = state_dict[f'soc_net.layer2']

        state_dict[f'soc_net.layer2'] = layer2
        state_dict[f'soc_net.layer1'] = layer1

    model_config.update({"proj_atom_anglr_m":proj_atom_anglr_m,
                            "skint_types":reducted_skint_types,
                            "onsite_num":onsite_num,
                            "soc":soc})

    return model_config, state_dict
