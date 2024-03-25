import numpy as np
import ase
from scipy.constants import Boltzmann, pi, elementary_charge, hbar
import torch

anglrMId = {'s':0,'p':1,'d':2,'f':3,'g':4,'h':5}
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

Orbital_Order_Wan_Default = { 's': ['s'],
                              'p': ['pz','px','py'],
                              'd': ['dz2','dxz','dyz','dx2-y2','dxy']}
Orbital_Order_SK = {'s': ['s'],
                    'p': ['py','pz','px'],
                    'd': ['dxy','dyz','dz2','dxz','dx2-y2']}

ABACUS_orbital_number_m = {
    "s": [0],
    "p": [0, 1, -1],
    "d": [0, 1, -1, 2, -2],
    "f": [0, 1, -1, 2, -2, 3, -3],
    "g": [0, 1, -1, 2, -2, 3, -3, 4, -4],
    "h": [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
}

DeePTB_orbital_number_m = {
    "s": [0],
    "p": [-1, 0, 1],
    "d": [-2, -1, 0, 1, 2],
    "f": [-3, -2, -1, 0, 1, 2, 3],
    "g": [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    "h": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
}


ABACUS2DeePTB = {
            0: torch.eye(1),
            1: torch.eye(3)[[2, 0, 1]],
            2: torch.eye(5)[[4, 2, 0, 1, 3]],
            3: torch.eye(7)[[6, 4, 2, 0, 1, 3, 5]],
            4: torch.eye(9)[[8, 6, 4, 2, 0, 1, 3, 5, 7]],
            5: torch.eye(11)[[10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9]]
        }
ABACUS2DeePTB[1][[0, 2]] *= -1
ABACUS2DeePTB[2][[1, 3]] *= -1
ABACUS2DeePTB[3][[0, 6, 2, 4]] *= -1
ABACUS2DeePTB[4][[1, 7, 3, 5]] *= -1
ABACUS2DeePTB[5][[0, 8, 2, 6, 4]] *= -1


dtype_dict = {"float32": torch.float32, "float64": torch.float64}
# k = Boltzmann # k is the Boltzmann constant in old NEGF module
Coulomb = 6.24150974e18 # in the unit of eV*Angstrom
eV2J = 1.6021766208e-19 # in the unit of J