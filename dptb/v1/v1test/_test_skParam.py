import os
import sys
import numpy as np
import pytest
import logging
import pickle
from dptb.utils.tools import get_uniq_symbol
from dptb.sktb.skParam import sk_init, interp_sk_gridvalues, read_skfiles

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
        return str(request.config.rootdir)


def test_sk_init(root_directory):
    sk_file_path = root_directory + '/examples/slakos'
    proj_atom_anglr_m = {'C':['s','p']}
    
    skfile = sk_init(proj_atom_anglr_m=proj_atom_anglr_m, sk_file_path=sk_file_path)
    assert len(skfile.keys()) == 1
    proj_atom_anglr_m = {'C':['s','p'],'H':['s']}
    skfiles = sk_init(proj_atom_anglr_m=proj_atom_anglr_m, sk_file_path=sk_file_path)
    assert len(skfiles.keys()) == 4

def test_read_skfiles(root_directory):
    sk_file_path = root_directory + '/examples/slakos'
    datafile = root_directory+'/dptb/tests/data/HSintgrl.pickle'
    fdat = open(datafile, 'rb')
    hsint=pickle.load(fdat)
    fdat.close()
    proj_atom_anglr_m = {'C': ['s', 'p'], 'H': ['s']}
    skfiles = sk_init(proj_atom_anglr_m=proj_atom_anglr_m, sk_file_path=sk_file_path)
    site_energy = {'C':[-5.0489172e-01, -1.9435511e-01,  0.0000000e+00],
                   'H':[-2.3860040e-01,  3.9000000e-05,  0.0000000e+00]}
    hubbard_u= {'C':[0.3647  , 0.387425, 0.341975],
                'H':[0.4195  , 0.4919  , 0.3471  ]}
    occupation = {'C':[2., 2., 0.],
                  'H':[1., 0., 0.]}
    
    grid_distance, num_grids, HSintgrl, SiteE, HubdUm, Occu = read_skfiles(skfiles=skfiles)
    for i in skfiles:
        assert (grid_distance[i] == 0.02), 'grid_distance is wrong'
        assert (num_grids[i] == 499), 'num_grids is wrong'
        assert (HSintgrl[i] == hsint[i]).all(), '{0} HSintgrl is wrong'.format(i)
    for i in ['C','H']:
        assert (SiteE[i] ==site_energy[i]).all(), 'SiteE is wrong'
        assert (HubdUm[i] == hubbard_u[i]).all(), 'HubdUm is wrong'
        assert (Occu[i] == occupation[i]).all(), 'Occu is wrong'


def test_interp_sk_gridvalues(root_directory):
    sk_file_path = root_directory + '/examples/slakos'
    proj_atom_anglr_m = {'C': ['s', 'p'], 'H': ['s']}
    skfiles = sk_init(proj_atom_anglr_m=proj_atom_anglr_m, sk_file_path=sk_file_path)
    datafile = root_directory+'/dptb/tests/data/HSintgrl.pickle'
    fdat = open(datafile, 'rb')
    hsint=pickle.load(fdat)
    fdat.close()
    max_mins = {'C-C': [10.979999000000001, 0.02],
                'C-H': [10.979999000000001, 0.02],
                'H-C': [10.979999000000001, 0.02],
                'H-H': [10.979999000000001, 0.02]}
    grid_distance, num_grids, HSintgrl, SiteE, HubdUm, Occu = read_skfiles(skfiles=skfiles)
    skfile_types = list(skfiles.keys())
    max_min_bond_length, interp_skfunc = interp_sk_gridvalues(skfile_types, grid_distance, num_grids, HSintgrl)
    x = np.arange(1,num_grids['C-C']+1)*grid_distance['C-C']
    ff=interp_skfunc['C-C'](x)
    ((ff - HSintgrl['C-C'])<1e-6).all()
    for i in skfiles:
        xx = np.arange(1,num_grids[i]+1)*grid_distance[i]
        fvalues =interp_skfunc[i](xx)
        assert (np.abs(fvalues - hsint[i])<1e-6).all()
        assert (np.abs(np.asarray(max_min_bond_length[i]) - np.asarray(max_mins[i])) < 1e-6).all()