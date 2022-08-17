from dptb.nnsktb.sknet import SKNet
import  torch

def test_SKnet():
    reducted_skint_types = ['N-N-2s-2s-0', 'N-B-2s-2p-0', 'B-B-2p-2p-0', 'B-B-2p-2p-1']
    nout = 4
    interneural = 10
    model = SKNet(reducted_skint_types, nout=nout, interneural=interneural)
    
    paras = list(model.parameters())
    assert len(paras) == 2
    assert paras[0].shape == torch.Size([len(reducted_skint_types), interneural])
    assert paras[1].shape == torch.Size([interneural, nout])
    
    coeff = model()
    assert len(coeff) == len(reducted_skint_types)

    for ikey in coeff.keys():
        assert ikey in reducted_skint_types
        assert coeff[ikey].shape == torch.Size([nout])


