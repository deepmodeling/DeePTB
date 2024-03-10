import pytest
import logging
import numpy as np
from dptb.sktb.struct_skhs import SKHSLists
from dptb.sktb.skIntegrals import SKIntegrals
from dptb.structure.structure import BaseStruct


hoppings = [np.array([-0.00050115, -0.00078644,  0.00088991, -0.00143416,  0.00011374]),
     np.array([-0.01799905,  0.02771481,  0.03792326, -0.00573914]),
     np.array([-0.28569007, -0.34098223,  0.24573991, -0.26678096,  0.11842841]),
     np.array([-0.00050115, -0.00078644,  0.00088991, -0.00143416,  0.00011374]),
     np.array([-0.01003859, -0.01506124,  0.01525194, -0.02153742,  0.00272391]),
     np.array([-0.01799905,  0.02771481,  0.03792326, -0.00573914]),
     np.array([-0.28569016, -0.34098233,  0.24573996, -0.26678101,  0.11842846]),
     np.array([-0.01799905,  0.02771481,  0.03792326, -0.00573914]),
     np.array([-0.01003859, -0.01506125,  0.01525194, -0.02153743,  0.00272391]),
     np.array([-0.00050115, -0.00078644,  0.00088991, -0.00143416,  0.00011374]),
     np.array([-0.00050115, -0.00078644,  0.00088991, -0.00143416,  0.00011374]),
     np.array([-0.28569013, -0.34098229,  0.24573994, -0.26678099,  0.11842844]),
     np.array([-0.01003859, -0.01506124,  0.01525194, -0.02153742,  0.00272391]),
     np.array([-0.00050115, -0.00078644,  0.00088991, -0.00143416,  0.00011374]),
     np.array([-0.00050115, -0.00078644,  0.00088991, -0.00143416,  0.00011374]),
     np.array([-0.04062027, -0.05191701,  0.06326006, -0.0118352 ]),
     np.array([-0.04062027, -0.05191701,  0.06326006, -0.0118352 ]),
     np.array([-0.04062027, -0.05191701,  0.06326006, -0.0118352 ])]

overlaps = [np.array([ 3.53079307e-04,  4.45624595e-04, -5.09991889e-04,  7.07969364e-04, -8.78069229e-05]),
    np.array([ 0.01159506, -0.02087616, -0.03556381,  0.00462289]),
    np.array([ 0.26658797,  0.33658656, -0.2903073 ,  0.32499413, -0.1442579 ]),
    np.array([ 3.53079412e-04,  4.45624785e-04, -5.09992113e-04,  7.07969757e-04,-8.78069527e-05]),
    np.array([ 0.00672879,  0.01087626, -0.01234273,  0.01936821, -0.00207514]),
    np.array([ 0.01159506, -0.02087616, -0.03556381,  0.00462289]),
    np.array([ 0.26658806,  0.33658665, -0.29030737,  0.32499417, -0.14425796]),
    np.array([ 0.01159506, -0.02087616, -0.03556381,  0.00462289]),
    np.array([ 0.00672879,  0.01087627, -0.01234273,  0.01936821, -0.00207514]),
    np.array([ 3.53079416e-04,  4.45624792e-04, -5.09992121e-04,  7.07969770e-04,-8.78069537e-05]),
    np.array([ 3.53079451e-04,  4.45624856e-04, -5.09992196e-04,  7.07969901e-04,-8.78069636e-05]),
    np.array([ 0.26658803,  0.33658662, -0.29030734,  0.32499415, -0.14425794]),
    np.array([ 0.00672878,  0.01087626, -0.01234272,  0.0193682 , -0.00207514]),
    np.array([ 3.53079287e-04,  4.45624559e-04, -5.09991848e-04,  7.07969292e-04,-8.78069174e-05]),
    np.array([ 3.53079358e-04,  4.45624687e-04, -5.09991997e-04,  7.07969554e-04,-8.78069372e-05]),
    np.array([ 0.03992768,  0.05856833, -0.08387583,  0.01268807]),
    np.array([ 0.03992768,  0.05856833, -0.08387583,  0.01268807]),
    np.array([ 0.03992768,  0.05856833, -0.08387583,  0.01268807])]

onsiteEs = [np.array([-0.671363, -0.261222]), np.array([-0.339811, -0.131903])]
onsiteSs = [np.array([1., 1.]), np.array([1., 1.])]


@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
        return str(request.config.rootdir)

def test_SKHSLists(root_directory):
    sk_file_path = root_directory + '/examples/slakos'
    filename = root_directory + '/dptb/tests/data/hBN/hBN.vasp'
    proj_atom_anglr_m = {"N":["s","p"],"B":["s","p"]}
    proj_atom_neles = {"N": 5,"B":3}
    CutOff = 4
    struct = BaseStruct(atom=filename,format='vasp',
        cutoff=CutOff,proj_atom_anglr_m=proj_atom_anglr_m,proj_atom_neles=proj_atom_neles)
    struct.get_bond()
    skint = SKIntegrals(proj_atom_anglr_m,sk_file_path)
    hslist = SKHSLists(skint,dtype='numpy')
    hslist.update_struct(struct)
    hslist.get_HS_list()

    assert len(hoppings) == len(hslist.hoppings)
    for i in range(len(hoppings)):
        assert (np.abs(hoppings[i] - hslist.hoppings[i]) < 1e-6).all(), '{0} is wrong'.format(i)

    assert len(onsiteEs) == len(hslist.onsiteEs)
    for i in range(len(onsiteEs)):
        assert (np.abs(onsiteEs[i] - hslist.onsiteEs[i]) < 1e-6).all(), '{0} is wrong'.format(i)
    
    assert len(overlaps) == len(hslist.overlaps)
    for i in range(len(overlaps)):
        assert (np.abs(overlaps[i] - hslist.overlaps[i]) < 1e-6).all(), '{0} is wrong'.format(i)
    
    assert len(onsiteSs) == len(hslist.onsiteSs)
    for i in range(len(onsiteSs)):
        assert (np.abs(onsiteSs[i] - hslist.onsiteSs[i]) < 1e-6).all(), '{0} is wrong'.format(i)