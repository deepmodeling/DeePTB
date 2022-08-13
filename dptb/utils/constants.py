import numpy as np
import ase

anglrMId = {'s':0,'p':1,'d':2,'f':3}
SKBondType = {0:'sigma',1:'pi',2:'delta'}
au2Ang = 0.529177249
# bond integral index in DFTB sk files. specific.
SKAnglrMHSID = {'dd':np.array([0,1,2]),
                'dp':np.array([3,4]), 'pd':np.array([3,4]),
                'pp':np.array([5,6]),
                'ds':np.array([7]),   'sd':np.array([7]),
                'ps':np.array([8]),   'sp':np.array([8]),
                'ss':np.array([9])}
h_all_types = ['ss','sp','sd','pp','pd','dd']

atomic_num_dict = ase.atom.atomic_numbers
atomic_num_dict_r = dict(zip(atomic_num_dict.values(), atomic_num_dict.keys()))
MaxShells  = 3
NumHvals   = 10
