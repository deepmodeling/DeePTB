from dptb.nnsktb.sknet import SKNet
from dptb.nnsktb.skintTypes import all_onsite_ene_types, all_onsite_intgrl_types, all_skint_types
import  torch
from dptb.utils.index_mapping import Index_Mapings
import pytest

class TestSKnet:    

    onsite_num = {'N':4,'B':3}
    bond_neurons = {'nhidden':5,'nout':4}
    bond_neurons_overlap = {'nhidden':5,'nout':4,"nout_overlap":2}
    onsite_neurons = {'nhidden':6,'nout':4}
    onsite_strian_neurons = {'nhidden':8,'nout':5}
    soc_neurons = {'nhidden':7}
                         
    proj_atom_anglr_m = {'B':['3s'],'N':['2s','2p']}
    indexmap = Index_Mapings(proj_atom_anglr_m) 
    bond_index_map, bond_num_hops = indexmap.Bond_Ind_Mapings()
    _, _, onsite_index_map, onsite_num = indexmap.Onsite_Ind_Mapings(onsitemode='uniform',atomtype=['N','B'])
    onsite_strain_index_map, onsite_strain_num, _, _ = indexmap.Onsite_Ind_Mapings(onsitemode='strain',atomtype=['N','B'])

    all_onsiteE_types_dict, reduced_onsiteE_types, onsiteE_ind_dict = all_onsite_ene_types(onsite_index_map)
    all_skint_types_dict, reducted_skint_types, sk_bond_ind_dict = all_skint_types(bond_index_map=bond_index_map)
    all_onsite_int_types_dict, reducted_onsiteint_types, sk_onsite_ind_dict = all_onsite_intgrl_types(onsite_strain_index_map)

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
    
    modelnrl = SKNet(skint_types=reducted_skint_types, onsite_types=reduced_onsiteE_types, soc_types=reduced_onsiteE_types, hopping_neurons=bond_neurons,
                    onsite_neurons=onsite_neurons, onsite_index_dict=onsiteE_ind_dict, onsitemode='NRL', overlap=False)
    modelnrl_overlap = SKNet(skint_types=reducted_skint_types, onsite_types= reduced_onsiteE_types, soc_types= reduced_onsiteE_types, hopping_neurons= bond_neurons_overlap,
                    onsite_neurons= onsite_neurons, onsite_index_dict= onsiteE_ind_dict, onsitemode='NRL', overlap=True)
        
    modelnrlsoc = SKNet(skint_types=reducted_skint_types, onsite_types=reduced_onsiteE_types, soc_types=reduced_onsiteE_types, hopping_neurons=bond_neurons,
                    onsite_neurons=onsite_neurons, soc_neurons=soc_neurons, onsite_index_dict=onsiteE_ind_dict, onsitemode='NRL', overlap=False)
    
    modelnrl_overlapsoc = SKNet(skint_types=reducted_skint_types, onsite_types= reduced_onsiteE_types, soc_types= reduced_onsiteE_types, hopping_neurons= bond_neurons_overlap,
                    onsite_neurons= onsite_neurons, soc_neurons= soc_neurons, onsite_index_dict= onsiteE_ind_dict, onsitemode='NRL', overlap=True)
        
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
        self._test_hopping(model = self.modelnrl)
        self._test_hopping(model = self.modelnrlsoc)

    def _test_hopping_nrl_overlap(self,model):
        with pytest.raises(KeyError) as exception_info:
            modelnrl_overlap = SKNet(skint_types=self.reducted_skint_types, onsite_types=self.reduced_onsiteE_types, soc_types=self.reduced_onsiteE_types, hopping_neurons=self.bond_neurons,
                    onsite_neurons=self.onsite_neurons, onsite_index_dict=self.onsiteE_ind_dict, onsitemode='NRL', overlap=True)

        paras = list(model.hopping_net.parameters())
        assert len(paras) == 2
        assert paras[0].shape == torch.Size([len(self.reducted_skint_types), 1, self.bond_neurons_overlap['nhidden']])
        assert paras[1].shape == torch.Size([len(self.reducted_skint_types), self.bond_neurons_overlap['nout']+self.bond_neurons_overlap['nout_overlap'], self.bond_neurons_overlap['nhidden']])

        coeff, ovelap_coeff = model(mode='hopping')
        assert len(coeff) == len(self.reducted_skint_types)
        assert  ovelap_coeff is not None
        assert len(ovelap_coeff) == len(self.reducted_skint_types)
        assert coeff.keys() == ovelap_coeff.keys()

        for ikey in coeff.keys():
            assert ikey in self.reducted_skint_types
            assert coeff[ikey].shape == torch.Size([self.bond_neurons_overlap['nout']])
            assert ovelap_coeff[ikey].shape == torch.Size([self.bond_neurons_overlap['nout_overlap']])

    def test_hopping_nrl_overlap(self):
        self._test_hopping_nrl_overlap(model = self.modelnrl_overlap)
        self._test_hopping_nrl_overlap(model = self.modelnrl_overlapsoc)


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
        assert paras[1].shape == torch.Size([len(self.reduced_onsiteE_types), self.onsite_neurons['nout'], self.onsite_neurons['nhidden']])

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

    def test_onsite_nrl(self):
            
        paras = list(self.modelnrl_overlap.onsite_net.parameters())
        assert len(paras) == 2
        assert paras[0].shape == torch.Size([len(self.reduced_onsiteE_types), 1, self.onsite_neurons['nhidden']])
        assert paras[1].shape == torch.Size([len(self.reduced_onsiteE_types), self.onsite_neurons['nout'], self.onsite_neurons['nhidden']])

        onsite_paras, _ = self.modelnrl_overlap(mode='onsite')
        assert isinstance(onsite_paras, dict)

        for ikey in onsite_paras.keys():
            assert ikey in self.reduced_onsiteE_types
            assert onsite_paras[ikey].shape == torch.Size([self.onsite_neurons['nout']])

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
        self._test_soc(model = self.modelnrlsoc)
        self._test_soc(model = self.modelnrl_overlapsoc)