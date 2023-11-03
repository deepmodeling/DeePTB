import torch
from e3nn.o3 import wigner_3j, Irrep, xyz_to_angles, Irrep
from dptb.utils.constants import h_all_types
from typing import Tuple

'''
The rotation matrix can be constructed as a batch form according to each orbital binding
'''


class RotationSE3(object):
    ''' rotate the SK parameters into the tight binding paras.

        Args:
            rot_type: 'tensor' use for torch tensor
                     'array' use for numpy array
        Attributes:
            function: rot_HS
                     rotate the SK paras $ss^ sigma$, $sp^ sigma$, $sd^ sigma$,
                                         $pp^ sigma$, $pp^ pi$, $pd^ sigma$, $pd^ pi$,
                                         $dd^ sigma$,$dd^ pi$,$dd^ delta$
                     into tight binding hoppings, according to the direction vector  rij/|rij|.
            function: ss sp sd pp pd dd :
                     define rotation functions.
    '''

    def __init__(self , rot_type, device) -> None:
        print('# initial rotate H or S func.')
        self.rot_type = rot_type
        self.device = device



        # self.sd = sd
        # self.pd = pd
        # self.dd = dd

    def rot_HS(self, Htype, Hvalue, Angvec):
        assert Htype in h_all_types, "Wrong hktypes"
        assert len(Hvalue.shape) in [1,2]
        assert len(Angvec.shape) in [1,2]

        if len(Hvalue.shape) == 1:
            Hvalue = Hvalue.unsqueeze(0)
        if len(Angvec.shape) == 1:
            Angvec = Angvec.unsqueeze(0)
        
        Hvalue = Hvalue.type(self.rot_type)
        Angvec = Angvec.type(self.rot_type)
        
        Angvec = Angvec[:,[1,2,0]]

        switch = {'ss': [0,0],
                  'sp': [0,1],
                  'sd': [0,2],
                  'pp': [1,1],
                  'pd': [1,2],
                  'dd': [2,2]}
        irs_index = {
            'ss': [0],
            'sp': [1],
            'sd': [2],
            'pp': [0,6],
            'pd': [1,11],
            'dd': [0,6,20]
        }

        transform = {
            'ss': torch.tensor([[1.]], dtype=self.rot_type, device=self.device),
            'sp': torch.tensor([[1.]], dtype=self.rot_type, device=self.device),
            'sd': torch.tensor([[1.]], dtype=self.rot_type, device=self.device),
            'pp': torch.tensor([
                [3**0.5/3,2/3*3**0.5],[6**0.5/3,-6**0.5/3]
            ], dtype=self.rot_type, device=self.device
            ),
            'pd':torch.tensor([
                [(2/5)**0.5,(6/5)**0.5],[(3/5)**0.5,-2/5**0.5]
            ], dtype=self.rot_type, device=self.device
            ),
            'dd':torch.tensor([
                [5**0.5/5, 2*5**0.5/5, 2*5**0.5/5],
                [2*(1/14)**0.5,2*(1/14)**0.5,-4*(1/14)**0.5],
                [3*(2/35)**0.5,-4*(2/35)**0.5,(2/35)**0.5]
                ], dtype=self.rot_type, device=self.device
            )
        }

        nirs = (2*switch[Htype][0]+1) * (2 * switch[Htype][1]+1)
        # handle the Hvalue's shape

        irs = torch.zeros(Hvalue.shape[0], nirs, dtype=self.rot_type, device=self.device)
        irs[:, irs_index[Htype]] = (transform[Htype] @ Hvalue.T).T

        hs = transform_o3(Angvec=Angvec, L_vec=switch[Htype], irs=irs, dtype=self.rot_type, device=self.device)
        hs = hs.transpose(1,2)

        if hs.shape[0] == 1:
            return hs.squeeze(0)
        return hs
    
    def rot_E3(self):
        pass


def transform_o3(Angvec: torch.Tensor, L_vec: Tuple, irs: torch.Tensor, dtype=torch.float64, device="cpu"):
    """_summary_

    Parameters
    ----------
    Angvec : torch.Tensor
        direction cosines of shift vector \hat{R}, in order [y,z,x].
    L_vec : torch.Tensor 
        looks like torch.tensor([l1, l2]), where l1 <= l2.
    irs : torch.Tensor
        the irreducible representation of operator block under basis of sperical harmonics
        denoted by l1 and l2.
    """
    assert len(irs.shape) in [1,2]
    assert len(Angvec.shape) in [1,2]
    assert len(L_vec) == 2
    
    if len(irs.shape) == 1:
        irs = irs.unsqueeze(0)
    if len(Angvec.shape) == 1:
        Angvec = Angvec.unsqueeze(0)

    l1, l2 = L_vec[0], L_vec[1]
    wms = []
    assert len(irs.reshape(-1)) == (2*l1+1) * (2*l2+1) * len(Angvec)
    for l_ird in range(abs(l2-l1), l2+l1+1):
        wms.append(wigner_3j(int(l1), int(l2), int(l_ird), dtype=dtype, device=device) * (2*l_ird+1)**0.5)
    
    wms = torch.cat(wms, dim=-1)
    H_ird = torch.sum(wms[None,:,:,:] * irs[:,None, None, :], dim=-1) # shape (N, 2l1+1, 2l2+1)


    angle = xyz_to_angles(Angvec) # (tensor(N), tensor(N))
    rot_mat_L = Irrep(int(l1), 1).D_from_angles(angle[0], angle[1], torch.tensor(0., dtype=dtype, device=device)) # tensor(N, 2l1+1, 2l1+1)
    rot_mat_R = Irrep(int(l2), 1).D_from_angles(angle[0], angle[1], torch.tensor(0., dtype=dtype, device=device)) # tensor(N, 2l2+1, 2l2+1)

    HR = rot_mat_L @ H_ird @ rot_mat_R.transpose(1,2)

    return HR




    
    
