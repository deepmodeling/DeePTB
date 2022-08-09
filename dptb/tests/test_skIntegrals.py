import pytest
import os
import pickle
import numpy as np
from dptb.sktb.skIntegrals import SKIntegrals

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    """
    :return:
    """
    return str(request.config.rootdir)


def test_SKIntegrals(root_directory):
    sk_file_path = root_directory + '/examples/slakos'
    proj_atom_anglr_m = {'C': ['s', 'p'], 'H': ['s']}
    skint = SKIntegrals(proj_atom_anglr_m, sk_file_path)
    datafile = root_directory+'/dptb/tests/data/HSintgrl.pickle'
    fdat = open(datafile, 'rb')
    hsint=pickle.load(fdat)
    fdat.close()
    skfiletype = ['C-C','C-H','H-C','H-H']  
    xx = np.arange(1,499+1)*0.02
    for isktype in skfiletype:
        atomtypes = isktype.split(sep='-')
        for i in range(10):
            ii = np.random.choice(498,1) + 1
            hsvalue = skint.sk_integral(itype=atomtypes[0], 
                                jtype=atomtypes[1], dist=xx[ii])
            assert (np.abs(hsvalue - hsint[isktype][ii])<1e-6).all(), '{0} is wrong'.format(isktype)



