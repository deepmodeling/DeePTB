from dptb.nnsktb.sknet import SKNet
from dptb.nnsktb.skintTypes import all_onsite_ene_types, all_onsite_intgrl_types, all_skint_types
import  torch
from dptb.utils.index_mapping import Index_Mapings

class TestSKnet:    

    reducted_skint_types = ['N-N-2s-2s-0', 'N-B-2s-2p-0', 'B-B-2p-2p-0', 'B-B-2p-2p-1']
    onsite_num = {'N':4,'B':3}
    bond_neurons = {'nhidden':5,'nout':4}
    onsite_neurons = {'nhidden':6}
    

    onsite_num2 = {'N':2,'B':1}

    reducted_onsiteint_types = ['N-N-2s-2s-0',
                                      'N-B-2s-2s-0',
                                      'N-B-2s-2p-0',
                                      'N-B-2p-2p-0',
                                      'N-B-2p-2p-1',
                                      'B-N-2s-2s-0',
                                      'B-B-2s-2s-0']

    proj_atom_anglr_m = {'B':['2s'],'N':['2s','2p']}
    indexmap = Index_Mapings(proj_atom_anglr_m) 
    bond_index_map, bond_num_hops = indexmap.Bond_Ind_Mapings()
    onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num = indexmap.Onsite_Ind_Mapings(onsitemode='uniform',atomtype=['N','B'])
    all_onsiteE_types_dict, reduced_onsiteE_types, onsiteE_ind_dict = all_onsite_ene_types(onsite_index_map)
    all_skint_types_dict, reducted_skint_types, sk_bond_ind_dict = all_skint_types(bond_index_map=bond_index_map)

    modelstrain = SKNet(skint_types=reducted_skint_types, onsite_types=reducted_onsiteint_types, soc_types=reduced_onsiteE_types, hopping_neurons=bond_neurons, 
                    onsite_neurons=onsite_neurons, onsite_index_dict=onsiteE_ind_dict, onsitemode='strain')

    modeluniform = SKNet(skint_types=reducted_skint_types, onsite_types=reduced_onsiteE_types, soc_types=reduced_onsiteE_types, hopping_neurons=bond_neurons, 
                    onsite_neurons=onsite_neurons,  onsite_index_dict=onsiteE_ind_dict, onsitemode='uniform')
    
    soc_neurons = {'nhidden':6}
    modelstrainsoc = SKNet(skint_types=reducted_skint_types,  onsite_types=reducted_onsiteint_types, soc_types=reduced_onsiteE_types, hopping_neurons=bond_neurons, onsite_neurons=onsite_neurons, 
                        soc_neurons=soc_neurons, onsite_index_dict=onsiteE_ind_dict, onsitemode='strain')
    modeluniformsoc = SKNet(skint_types=reducted_skint_types, onsite_types=reduced_onsiteE_types, soc_types=reduced_onsiteE_types, hopping_neurons=bond_neurons, 
                    onsite_neurons=onsite_neurons,soc_neurons=soc_neurons, onsite_index_dict=onsiteE_ind_dict, onsitemode='uniform')

    

    bond_neurons = {'nhidden':5,'nout':4}
    onsite_neurons = {'nhidden':6}



    def test_bond(self):

        paras = list(self.modeluniform.hopping_net.parameters())
        assert len(paras) == 2
        assert paras[0].shape == torch.Size([len(self.reducted_skint_types), self.bond_neurons['nhidden']])
        assert paras[1].shape == torch.Size([self.bond_neurons['nhidden'], self.bond_neurons['nout']])

        coeff = self.modeluniform(mode='hopping')
        assert len(coeff) == len(self.reducted_skint_types)

        for ikey in coeff.keys():
            assert ikey in self.reducted_skint_types
            assert coeff[ikey].shape == torch.Size([self.bond_neurons['nout']])

    def test_onsite_uniform(self):
        sknet = SKNet(skint_types=self.reducted_skint_types,onsite_types=self.reduced_onsiteE_types,soc_types=self.reduced_onsiteE_types,onsite_index_dict=self.onsiteE_ind_dict,
                        hopping_neurons=self.bond_neurons, onsite_neurons=self.onsite_neurons, onsitemode='uniform')
        onsite_values, _ = sknet(mode = 'onsite')

        assert onsite_values['N'].shape == torch.Size([2])
        assert onsite_values['B'].shape == torch.Size([1])


    def test_onsite_strain(self):

        paras = list(self.modelstrain.onsite_net.parameters())
        assert len(paras) == 2
        assert paras[0].shape == torch.Size([len(self.reducted_onsiteint_types), self.bond_neurons['nhidden']])
        assert paras[1].shape == torch.Size([self.bond_neurons['nhidden'], self.bond_neurons['nout']])

        _, coeff = self.modelstrain(mode='onsite')
        assert len(coeff) == len(self.reducted_onsiteint_types)
 
        for ikey in coeff.keys():
            assert ikey in self.reducted_onsiteint_types
            assert coeff[ikey].shape == torch.Size([self.bond_neurons['nout']])

    # def test_onsite_uniform_soc(self):

    #     paras = list(self.modeluniformsoc.onsite_net.parameters())
    #     assert len(paras) == 4
    #     for ia in self.onsite_num2:
    #         paras = list(self.modeluniformsoc.onsite_net[ia].parameters())
    #         assert len(paras) == 2
    #         assert paras[0].shape == torch.Size([1, self.onsite_neurons['nhidden']])
    #         assert paras[1].shape == torch.Size([self.onsite_neurons['nhidden'],self.onsite_num2[ia]])



    #     paras = list(self.modeluniformsoc.soc_net.parameters())
    #     assert len(paras) == 4
    #     for ia in self.onsite_num2:
    #         paras = list(self.modeluniformsoc.soc_net[ia].parameters())
    #         assert len(paras) == 2
    #         assert paras[0].shape == torch.Size([1, self.soc_neurons['nhidden']])
    #         assert paras[1].shape == torch.Size([self.soc_neurons['nhidden'], self.onsite_num2[ia]])
    
    # def test_onsite_strain_soc(self):
        
    #     paras = list(self.modelstrainsoc.onsite_net.parameters())
    #     assert len(paras) == 2
    #     assert paras[0].shape == torch.Size([len(self.reducted_onsiteint_types), self.bond_neurons['nhidden']])
    #     assert paras[1].shape == torch.Size([self.bond_neurons['nhidden'], self.bond_neurons['nout']])

    #     _, coeff = self.modelstrainsoc(mode='onsite')
    #     assert len(coeff) == len(self.reducted_onsiteint_types)
 
    #     for ikey in coeff.keys():
    #         assert ikey in self.reducted_onsiteint_types
    #         assert coeff[ikey].shape == torch.Size([self.bond_neurons['nout']])


        # paras = list(self.modelstrainsoc.soc_net.parameters())
        # assert len(paras) == 4
        # for ia in self.onsite_num2:
        #     paras = list(self.modelstrainsoc.soc_net[ia].parameters())
        #     assert len(paras) == 2
        #     assert paras[0].shape == torch.Size([1, self.soc_neurons['nhidden']])
        #     assert paras[1].shape == torch.Size([self.soc_neurons['nhidden'], self.onsite_num2[ia]])
