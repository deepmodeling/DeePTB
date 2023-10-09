from dptb.nnsktb.sknet import SKNet
from dptb.nnsktb.skintTypes import all_onsite_ene_types, all_onsite_intgrl_types, all_skint_types
import  torch
from dptb.utils.index_mapping import Index_Mapings
import pytest

class TestSKnet:    

    reducted_skint_types = ['N-N-2s-2s-0', 'N-B-2s-2p-0', 'B-B-2p-2p-0', 'B-B-2p-2p-1']
    onsite_num = {'N':4,'B':3}
    bond_neurons = {'nhidden':5,'nout':4}
    onsite_neurons = {'nhidden':6,'nout':1}
    onsite_strian_neurons = {'nhidden':8,'nout':5}
    soc_neurons = {'nhidden':6}
                         
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
    _, _, onsite_index_map, onsite_num = indexmap.Onsite_Ind_Mapings(onsitemode='uniform',atomtype=['N','B'])
    all_onsiteE_types_dict, reduced_onsiteE_types, onsiteE_ind_dict = all_onsite_ene_types(onsite_index_map)
    all_skint_types_dict, reducted_skint_types, sk_bond_ind_dict = all_skint_types(bond_index_map=bond_index_map)

    modelstrain = SKNet(skint_types=reducted_skint_types, onsite_types=reducted_onsiteint_types, soc_types=reduced_onsiteE_types, hopping_neurons=bond_neurons, 
                    onsite_neurons=onsite_strian_neurons, onsite_index_dict=onsiteE_ind_dict, onsitemode='strain')

    modeluniform = SKNet(skint_types=reducted_skint_types, onsite_types=reduced_onsiteE_types, soc_types=reduced_onsiteE_types, hopping_neurons=bond_neurons, 
                    onsite_neurons=onsite_neurons,  onsite_index_dict=onsiteE_ind_dict, onsitemode='uniform')
    
    modelnone= SKNet(skint_types=reducted_skint_types, onsite_types=reduced_onsiteE_types, soc_types=reduced_onsiteE_types, hopping_neurons=bond_neurons, 
                    onsite_neurons=onsite_neurons,  onsite_index_dict=onsiteE_ind_dict, onsitemode='none')
    

    modelstrainsoc = SKNet(skint_types=reducted_skint_types,  onsite_types=reducted_onsiteint_types, soc_types=reduced_onsiteE_types, hopping_neurons=bond_neurons, 
                    onsite_neurons=onsite_strian_neurons, soc_neurons=soc_neurons, onsite_index_dict=onsiteE_ind_dict, onsitemode='strain')
    modeluniformsoc = SKNet(skint_types=reducted_skint_types, onsite_types=reduced_onsiteE_types, soc_types=reduced_onsiteE_types, hopping_neurons=bond_neurons, 
                    onsite_neurons=onsite_neurons,soc_neurons=soc_neurons, onsite_index_dict=onsiteE_ind_dict, onsitemode='uniform')
    modelnonesoc= SKNet(skint_types=reducted_skint_types, onsite_types=reduced_onsiteE_types, soc_types=reduced_onsiteE_types, hopping_neurons=bond_neurons, 
                    onsite_neurons=onsite_neurons, soc_neurons=soc_neurons, onsite_index_dict=onsiteE_ind_dict, onsitemode='none')
    

    def _test_hopping(self,model):

        paras = list(model.hopping_net.parameters())
        assert len(paras) == 2
        assert paras[0].shape == torch.Size([len(self.reducted_skint_types), 1, self.bond_neurons['nhidden']])
        assert paras[1].shape == torch.Size([len(self.reducted_skint_types), self.bond_neurons['nout'], self.bond_neurons['nhidden']])

        coeff, ovelap_coeff = model(mode='hopping')
        assert len(coeff) == len(self.reducted_skint_types)
        if ovelap_coeff is not None:
            assert len(ovelap_coeff) == len(self.reducted_skint_types)
        for ikey in coeff.keys():
            assert ikey in self.reducted_skint_types
            assert coeff[ikey].shape == torch.Size([self.bond_neurons['nout']])

    def test_hopping(self):
        self._test_hopping(model = self.modeluniform)
        self._test_hopping(model = self.modelnone)
        self._test_hopping(model = self.modelstrain)
        self._test_hopping(model = self.modeluniformsoc)
        self._test_hopping(model = self.modelstrainsoc)
        self._test_hopping(model = self.modelnonesoc)


    def test_onsite_none(self):
        with pytest.raises(AttributeError) as exception_info:
            self.modelnone.onsite_net()
        
        onsite_values, coeff = self.modelnone(mode = 'onsite')
        assert onsite_values is None
        assert coeff is None


    def test_onsite_uniform(self):
        paras = list(self.modeluniform.onsite_net.parameters())
        assert len(paras) == 2
        assert paras[0].shape == torch.Size([len(self.reduced_onsiteE_types), 1, self.onsite_neurons['nhidden']])
        assert paras[1].shape == torch.Size([len(self.reduced_onsiteE_types), 1, self.onsite_neurons['nhidden']])

        #sknet = SKNet(skint_types=self.reducted_skint_types,onsite_types=self.reduced_onsiteE_types,soc_types=self.reduced_onsiteE_types,onsite_index_dict=self.onsiteE_ind_dict,
        #                hopping_neurons=self.bond_neurons, onsite_neurons=self.onsite_neurons, onsitemode='uniform')
        onsite_values, _ = self.modeluniform(mode = 'onsite')

        assert isinstance(onsite_values, dict)
        assert onsite_values['N'].shape == torch.Size([2])
        assert onsite_values['B'].shape == torch.Size([1])


    def test_onsite_strain(self):

        paras = list(self.modelstrain.onsite_net.parameters())
        assert len(paras) == 2
        assert paras[0].shape == torch.Size([len(self.reducted_onsiteint_types), 1, self.onsite_strian_neurons['nhidden']])
        assert paras[1].shape == torch.Size([len(self.reducted_onsiteint_types), self.onsite_strian_neurons['nout'], self.onsite_strian_neurons['nhidden']])

        _, coeff = self.modelstrain(mode='onsite')
        assert len(coeff) == len(self.reducted_onsiteint_types)
 
        for ikey in coeff.keys():
            assert ikey in self.reducted_onsiteint_types
            assert coeff[ikey].shape == torch.Size([self.onsite_strian_neurons['nout']])

    def _test_soc(self, model):
        paras = list(model.soc_net.parameters())
        assert len(paras) == 2
        assert paras[0].shape == torch.Size([len(self.reduced_onsiteE_types), 1, self.soc_neurons['nhidden']])
        assert paras[1].shape == torch.Size([len(self.reduced_onsiteE_types), 1, self.soc_neurons['nhidden']])

        soc_value,_ = model(mode='soc')
        assert isinstance(soc_value, dict)
        assert soc_value['N'].shape == torch.Size([2])
        assert soc_value['B'].shape == torch.Size([1])

    def test_soc(self):
        self._test_soc(model = self.modeluniformsoc)
        self._test_soc(model = self.modelstrainsoc)
        self._test_soc(model = self.modelnonesoc)
    