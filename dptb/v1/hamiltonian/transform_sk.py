import numpy as np
import torch as th
from dptb.utils.constants import h_all_types

'''
The rotation matrix can be constructed as a batch form according to each orbital binding
'''

class RotationSK(object):
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

    def rot_HS(self, Htype, Hvalue, Angvec):
        assert Htype in h_all_types, "Wrong hktypes"

        switch = {'ss': self.ss,
                  'sp': self.sp,
                  'sd': self.sd,
                  'pp': self.pp,
                  'pd': self.pd,
                  'dd': self.dd}

        hs = switch.get(Htype)(Angvec, Hvalue)
        return hs


    def ss(self, Angvec, SKss):
        ## ss orbital no angular dependent.
        if not isinstance(SKss, th.Tensor):
            SKss = th.tensor(SKss, dtype=self.rot_type, device=self.device)
        SKss_rsp = th.reshape(SKss ,[1 ,1])
        hs  = SKss_rsp
        return hs

    def sp(self, Angvec, SKsp):
        if not isinstance(SKsp, th.Tensor):
            SKsp = th.tensor(SKsp, dtype=self.rot_type, device=self.device)

        x = Angvec[0]
        y = Angvec[1]
        z = Angvec[2]

        rot_mat = th.tensor([y ,z ,x], dtype=self.rot_type, device=self.device)
        rot_mat = rot_mat.reshape([3 ,1])
        # [3,1]*[1,1] => [3,1]
        SKsp_rsp = th.reshape(SKsp ,[1 ,1])
        rot_mat.requires_grad = False
        hs = th.matmul(rot_mat, SKsp_rsp)
        hs = th.reshape(hs ,[3 ,1])

        return hs

    def sd(self, Angvec, SKsd):
        if not isinstance(SKsd, th.Tensor):
            SKsd = th.tensor(SKsd, dtype=self.rot_type, device=self.device)
        x = Angvec[0]
        y = Angvec[1]
        z = Angvec[2]

        s3 = th.sqrt(th.scalar_tensor(3.0, dtype=self.rot_type, device=self.device))
        rot_mat = th.tensor([s3 * x * y, s3 * y *z, 1.5 * z**2 -0.5, \
                            s3 * x * z, s3 * (2.0 * x ** 2 - 1.0 + z ** 2) / 2.0], dtype=self.rot_type, device=self.device)
        rot_mat = rot_mat.reshape([5, 1])
        # [5,1]*[1,1] => [5,1]
        SKsd_rsp = th.reshape(SKsd, [1, 1])
        rot_mat.requires_grad = False
        hs = th.matmul(rot_mat, SKsd_rsp)
        hs = th.reshape(hs, [5, 1])

        return hs

    def pp(self, Angvec, SKpp):
        if not isinstance(SKpp, th.Tensor):
            SKpp = th.tensor(SKpp, dtype=self.rot_type, device=self.device)

        x = Angvec[0]
        y = Angvec[1]
        z = Angvec[2]
        rot_mat = th.zeros([3, 3, 2], dtype=self.rot_type, device=self.device)
        # [3,3,2] dot (2,1) => [3,3,1] => [3,3]
        rot_mat[0, 0, :] = th.tensor([1.0 - z ** 2 - x ** 2, z ** 2 + x ** 2], dtype=self.rot_type, device=self.device)
        rot_mat[0, 1, :] = th.tensor([z * y, - z * y], dtype=self.rot_type, device=self.device)
        rot_mat[0, 2, :] = th.tensor([x * y, - x * y], dtype=self.rot_type, device=self.device)
        rot_mat[1, 0, :] = rot_mat[0, 1, :]
        rot_mat[1, 1, :] = th.tensor([z ** 2, 1.0 - z ** 2], dtype=self.rot_type, device=self.device)
        rot_mat[1, 2, :] = th.tensor([z * x, - z * x], dtype=self.rot_type, device=self.device)
        rot_mat[2, 0, :] = rot_mat[0, 2, :]
        rot_mat[2, 1, :] = rot_mat[1, 2, :]
        rot_mat[2, 2, :] = th.tensor([x ** 2, 1 - x ** 2], dtype=self.rot_type, device=self.device)

        SKpp_rsp = th.reshape(SKpp, [2, 1])
        rot_mat.requires_grad = False
        hs = th.matmul(rot_mat, SKpp_rsp)
        hs = th.reshape(hs, [3, 3])

        return hs

    def pd(self, Angvec, SKpd):
        if not isinstance(SKpd, th.Tensor):
            SKpd = th.tensor(SKpd, dtype=self.rot_type, device=self.device)

        x = Angvec[0]
        y = Angvec[1]
        z = Angvec[2]
        s3 = th.sqrt(th.scalar_tensor(3.0, dtype=self.rot_type, device=self.device))
        rot_mat = th.zeros([5, 3, 2], dtype=self.rot_type, device=self.device)
        rot_mat[0, 0, :] = th.tensor([-(-1.0 + z ** 2 + x ** 2) * x * s3,
                                     (2.0 * z ** 2 + 2.0 * x ** 2 - 1.0) * x], dtype=self.rot_type, device=self.device)

        rot_mat[1, 0, :] = th.tensor([-(-1 + z ** 2 + x ** 2) * z * s3,
                                     (2 * z ** 2 + 2 * x ** 2 - 1.0) * z], dtype=self.rot_type, device=self.device)

        rot_mat[2, 0, :] = th.tensor([y * (3 * z ** 2 - 1) / 2.0, -s3 * z ** 2 * y], dtype=self.rot_type, device=self.device)

        rot_mat[3, 0, :] = th.tensor([s3 * y * x * z, -2 * x * y * z], dtype=self.rot_type, device=self.device)

        rot_mat[4, 0, :] = th.tensor([y * s3 * (2 * x ** 2 - 1 + z ** 2) / 2.0,
                                     -(z ** 2 + 2 * x ** 2) * y], dtype=self.rot_type, device=self.device)

        rot_mat[0, 1, :] = th.tensor([x * y * z * s3, -2.0 * z * x * y], dtype=self.rot_type, device=self.device)

        rot_mat[1, 1, :] = th.tensor([y * (z ** 2) * s3, -(2.0 * z ** 2 - 1.0) * y], dtype=self.rot_type, device=self.device)

        rot_mat[2, 1, :] = th.tensor([(z * (3.0 * z ** 2 - 1.0)) / 2.0,
                                     -z * s3 * (-1.0 + z ** 2)], dtype=self.rot_type, device=self.device)

        rot_mat[3, 1, :] = th.tensor([x * z ** 2 * s3, -1 * (2.0 * z ** 2 - 1.0) * x], dtype=self.rot_type, device=self.device)

        rot_mat[4, 1, :] = th.tensor([(2.0 * x ** 2 - 1.0 + z ** 2) * z * s3 / 2.0,
                                     -1 * (z * (2.0 * x ** 2 - 1.0 + z ** 2))], dtype=self.rot_type, device=self.device)

        rot_mat[0, 2, :] = th.tensor([(x ** 2) * y * s3, -1 * (2.0 * x ** 2 - 1.0) * y], dtype=self.rot_type, device=self.device)

        rot_mat[1, 2, :] = th.tensor([x * y * s3 * z, -2.0 * y * x * z], dtype=self.rot_type, device=self.device)

        rot_mat[2, 2, :] = th.tensor([(x * (3.0 * z ** 2 - 1.0)) / 2.0,
                                     -1 * s3 * (z ** 2) * x], dtype=self.rot_type, device=self.device)

        rot_mat[3, 2, :] = th.tensor([x ** 2 * s3 * z, -1 * (2.0 * x ** 2 - 1.0) * z], dtype=self.rot_type, device=self.device)

        rot_mat[4, 2, :] = th.tensor([x * (2.0 * x ** 2 - 1.0 + z ** 2) * s3 / 2.0,
                                     -1 * ((z ** 2 - 2.0 + 2.0 * x ** 2) * x)], dtype=self.rot_type, device=self.device)

        # [5,3,2] dot (2,1) => [5,3,1] => [5,3]
        SKpd_rsp = th.reshape(SKpd, [2, 1])
        rot_mat.requires_grad = False
        hs = th.matmul(rot_mat, SKpd_rsp)
        hs = th.reshape(hs, [5, 3])

        return hs

    def dd(self, Angvec, SKdd):
        if not isinstance(SKdd, th.Tensor):
            SKdd = th.tensor(SKdd, dtype=self.rot_type, device=self.device)

        x = Angvec[0]
        y = Angvec[1]
        z = Angvec[2]
        s3 = th.sqrt(th.scalar_tensor(3.0, dtype=self.rot_type, device=self.device))
        rot_mat = th.zeros([5, 5, 3], dtype=self.rot_type, device=self.device)

        rot_mat[0, 0, :] = th.tensor([-3.0 * x ** 2 * (-1.0 + z ** 2 + x ** 2),
                                     (4.0 * x ** 2 * z ** 2 - z ** 2 + 4.0 * x ** 4 - 4.0 * x ** 2 + 1.0),
                                     (-x ** 2 * z ** 2 + z ** 2 + x ** 2 - x ** 4)], dtype=self.rot_type, device=self.device)

        rot_mat[1, 0, :] = th.tensor([-3.0 * x * (-1.0 + z ** 2 + x ** 2) * z,
                                     (4.0 * z ** 2 + 4.0 * x ** 2 - 3.0) * z * x,
                                     -x * (z ** 2 + x ** 2) * z], dtype=self.rot_type, device=self.device)

        rot_mat[2, 0, :] = th.tensor([x * y * s3 * (3.0 * z ** 2 - 1.0) / 2.0,
                                     -2.0 * s3 * y * x * (z ** 2),
                                     x * y * (z ** 2 + 1.0) * s3 / 2.0], dtype=self.rot_type, device=self.device)

        rot_mat[3, 0, :] = th.tensor([3.0 * (x ** 2) * y * z,
                                     -(4.0 * x ** 2 - 1.0) * z * y,
                                     y * (-1.0 + x ** 2) * z], dtype=self.rot_type, device=self.device)

        rot_mat[4, 0, :] = th.tensor([3.0 / 2.0 * y * x * (2.0 * x ** 2 - 1.0 + z ** 2),
                                     -2.0 * y * x * (2.0 * x ** 2 - 1.0 + z ** 2),
                                     y * x * (2.0 * x ** 2 - 1.0 + z ** 2) / 2.0], dtype=self.rot_type, device=self.device)

        rot_mat[0, 1, :] = rot_mat[1, 0, :]

        rot_mat[1, 1, :] = th.tensor([-3.0 * (-1.0 + z ** 2 + x ** 2) * z ** 2,
                                     (4.0 * z ** 4 - 4.0 * z ** 2 + 4.0 * x ** 2 * z ** 2 + 1.0 - x ** 2),
                                     -(-1.0 + z) * (z ** 3 + z ** 2 + x ** 2 * z + x ** 2)], dtype=self.rot_type, device=self.device)

        rot_mat[2, 1, :] = th.tensor([y * s3 * z * (3.0 * z ** 2 - 1.0) / 2.0,
                                     -z * s3 * (2.0 * z ** 2 - 1.0) * y,
                                     (-1.0 + z ** 2) * s3 * z * y / 2.0], dtype=self.rot_type, device=self.device)

        rot_mat[3, 1, :] = th.tensor([3.0 * y * x * (z ** 2),
                                     -(4.0 * z ** 2 - 1.0) * y * x,
                                     x * y * (-1.0 + z ** 2)], dtype=self.rot_type, device=self.device)

        rot_mat[4, 1, :] = th.tensor([3.0 / 2.0 * y * (2.0 * x ** 2 - 1.0 + z ** 2) * z,
                                     -(2.0 * z ** 2 - 1.0 + 4.0 * x ** 2) * z * y,
                                     y * (z ** 2 + 2.0 * x ** 2 + 1.0) * z / 2.0], dtype=self.rot_type, device=self.device)

        rot_mat[0, 2, :] = rot_mat[2, 0, :]
        rot_mat[1, 2, :] = rot_mat[2, 1, :]

        rot_mat[2, 2, :] = th.tensor([((3.0 * z ** 2 - 1.0) ** 2) / 4.0,
                                     -(3.0 * (-1.0 + z ** 2) * z ** 2),
                                     3.0 / 4.0 * ((-1.0 + z ** 2) ** 2)], dtype=self.rot_type, device=self.device)

        rot_mat[3, 2, :] = th.tensor([x * (3.0 * z ** 2 - 1.0) * s3 * z / 2.0,
                                     -(2.0 * z ** 2 - 1.0) * x * z * s3,
                                     z * x * s3 * (-1.0 + z ** 2) / 2.0], dtype=self.rot_type, device=self.device)

        rot_mat[4, 2, :] = th.tensor([(2.0 * x ** 2 - 1.0 + z ** 2) * (3.0 * z ** 2 - 1.0) * s3 / 4.0,
                                     -(2.0 * x ** 2 - 1.0 + z ** 2) * (z ** 2) * s3,
                                     s3 * (2.0 * x ** 2 - 1.0 + z ** 2) * (z ** 2 + 1.0) / 4.0], dtype=self.rot_type, device=self.device)

        rot_mat[0, 3, :] = rot_mat[3, 0, :]
        rot_mat[1, 3, :] = rot_mat[3, 1, :]
        rot_mat[2, 3, :] = rot_mat[3, 2, :]

        rot_mat[3, 3, :] = th.tensor([3.0 * x ** 2 * z ** 2,
                                     (-4.0 * x ** 2 * z ** 2 + z ** 2 + x ** 2),
                                     (-1.0 + z) * (-z + x ** 2 * z - 1.0 + x ** 2)], dtype=self.rot_type, device=self.device)

        rot_mat[4, 3, :] = th.tensor([3.0 / 2.0 * x * (2.0 * x ** 2 - 1.0 + z ** 2) * z,
                                     -((2.0 * z ** 2 - 3.0 + 4.0 * x ** 2) * z * x),
                                     (x * (z ** 2 - 3.0 + 2.0 * x ** 2) * z) / 2.0], dtype=self.rot_type, device=self.device)

        rot_mat[0, 4, :] = rot_mat[4, 0, :]
        rot_mat[1, 4, :] = rot_mat[4, 1, :]
        rot_mat[2, 4, :] = rot_mat[4, 2, :]
        rot_mat[3, 4, :] = rot_mat[4, 3, :]
        rot_mat[4, 4, :] = th.tensor([3.0 / 4.0 * ((2.0 * x ** 2 - 1.0 + z ** 2) ** 2),
                                     -z ** 4 + z ** 2 - 4.0 * x ** 2 * z ** 2 - 4.0 * x ** 4 + 4.0 * x ** 2,
                                     (z ** 4 / 4.0 + x ** 2 * z ** 2 + z ** 2 / 2.0 + 1.0 / 4.0 - x ** 2 + x ** 4)], dtype=self.rot_type, device=self.device)

        # [5,5,3] dot (3,1) => [5,5,1] => [5,5]
        SKdd_rsp = th.reshape(SKdd, [3, 1])
        rot_mat.requires_grad = False
        hs = th.matmul(rot_mat, SKdd_rsp)
        hs = th.reshape(hs, [5, 5])

        return hs

