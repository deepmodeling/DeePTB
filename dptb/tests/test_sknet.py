from dptb.nnsktb.sknet import SKNet
import  torch

class TestSKnet:    

    reducted_skint_types = ['N-N-2s-2s-0', 'N-B-2s-2p-0', 'B-B-2p-2p-0', 'B-B-2p-2p-1']
    onsite_num = {'N':4,'B':3}
    bond_neurons = {'nhidden':5,'nout':4}
    onsite_neurons = {'nhidden':6}
    modeluniform = SKNet(skint_types=reducted_skint_types, onsite_num=onsite_num, bond_neurons=bond_neurons, 
                    onsite_neurons=onsite_neurons,onsitemode='uniform')

    onsite_num2 = {'N':2,'B':1}

    reducted_onsiteint_types = ['N-N-2s-2s-0',
                                      'N-B-2s-2s-0',
                                      'N-B-2s-2p-0',
                                      'N-B-2p-2p-0',
                                      'N-B-2p-2p-1',
                                      'B-N-2s-2s-0',
                                      'B-B-2s-2s-0']

    modelstrain = SKNet(skint_types=reducted_skint_types, onsite_num=onsite_num2, bond_neurons=bond_neurons, 
                    onsite_neurons=onsite_neurons, onsiteint_types=reducted_onsiteint_types, onsitemode='strain')
    
    soc_neurons = {'nhidden':6}
    modelsoc = SKNet(skint_types=reducted_skint_types, onsite_num=onsite_num2, bond_neurons=bond_neurons, onsite_neurons=onsite_neurons, 
                        soc_neurons=soc_neurons, onsiteint_types=reducted_onsiteint_types, onsitemode='soc')
    def test_bond(self):

        paras = list(self.modeluniform.bond_net.parameters())
        assert len(paras) == 2
        assert paras[0].shape == torch.Size([len(self.reducted_skint_types), self.bond_neurons['nhidden']])
        assert paras[1].shape == torch.Size([self.bond_neurons['nhidden'], self.bond_neurons['nout']])

        coeff = self.modeluniform(mode='hopping')
        assert len(coeff) == len(self.reducted_skint_types)

        for ikey in coeff.keys():
            assert ikey in self.reducted_skint_types
            assert coeff[ikey].shape == torch.Size([self.bond_neurons['nout']])

    def test_onsite_uniform(self):

        paras = list(self.modeluniform.onsite_net.parameters())
        assert len(paras) == 4
        for ia in self.onsite_num:
            paras = list(self.modeluniform.onsite_net[ia].parameters())
            assert len(paras) == 2
            assert paras[0].shape == torch.Size([1, self.onsite_neurons['nhidden']])
            assert paras[1].shape == torch.Size([self.onsite_neurons['nhidden'],self.onsite_num[ia]])


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

    def test_onsite_soc(self):

        paras = list(self.modelsoc.onsite_net.parameters())
        assert len(paras) == 4
        for ia in self.onsite_num:
            paras = list(self.modelsoc.onsite_net[ia].parameters())
            assert len(paras) == 2
            assert paras[0].shape == torch.Size([1, self.soc_neurons['nhidden']])
            assert paras[1].shape == torch.Size([self.soc_neurons['nhidden'],self.onsite_num2[ia]])
