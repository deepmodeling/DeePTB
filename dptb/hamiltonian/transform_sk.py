import numpy as np
import torch as th
from dptb.utils.constants import h_all_types


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

    def __init__(self ,rot_type) -> None:
        print('# initial rotate H or S func.')
        self.rot_type = rot_type.lower()

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
        if self.rot_type == 'tensor':
            SKss_rsp = th.reshape(SKss ,[1 ,1])
        else:
            SKss_rsp = np.reshape(SKss ,[1 ,1])

        hs  = SKss_rsp
        return hs

    def sp(self, Angvec, SKsp):
        x = Angvec[0]
        y = Angvec[1]
        z = Angvec[2]

        rot_mat = np.array([y ,z ,x])
        rot_mat = np.reshape(rot_mat ,[3 ,1])
        # [3,1]*[1,1] => [3,1]
        if self.rot_type == 'tensor':
            SKsp_rsp = th.reshape(SKsp ,[1 ,1])
            rot_mat_th = th.from_numpy(rot_mat).float()
            rot_mat_th.requires_grad = False
            hs = th.matmul(rot_mat_th, SKsp_rsp)
            hs = th.reshape(hs ,[3 ,1])
        else:
            SKsp_rsp = np.reshape(SKsp ,[1 ,1])
            hs = np.dot(rot_mat ,SKsp_rsp)
            hs = np.reshape(hs ,[3 ,1])

        return hs

    def sd(self, Angvec, SKsd):
        x = Angvec[0]
        y = Angvec[1]
        z = Angvec[2]

        s3 = np.sqrt(3.0)
        rot_mat = np.array([s3 * x *y, s3 * y *z, 1.5 * z**2 -0.5, \
                            s3 * x * z, s3 * (2.0 * x ** 2 - 1.0 + z ** 2) / 2.0])
        rot_mat = np.reshape(rot_mat, [5, 1])
        # [5,1]*[1,1] => [5,1]
        if self.rot_type == 'tensor':
            SKsd_rsp = th.reshape(SKsd, [1, 1])
            rot_mat_th = th.from_numpy(rot_mat).float()
            rot_mat_th.requires_grad = False
            hs = th.matmul(rot_mat_th, SKsd_rsp)
            hs = th.reshape(hs, [5, 1])
        else:
            SKsd_rsp = np.reshape(SKsd, [1, 1])
            hs = np.dot(rot_mat, SKsd_rsp)
            hs = np.reshape(hs, [5, 1])

        return hs

    def pp(self, Angvec, SKpp):
        x = Angvec[0]
        y = Angvec[1]
        z = Angvec[2]
        rot_mat = np.zeros([3, 3, 2])
        # [3,3,2] dot (2,1) => [3,3,1] => [3,3]
        rot_mat[0, 0, :] = np.array([1.0 - z ** 2 - x ** 2, z ** 2 + x ** 2])
        rot_mat[0, 1, :] = np.array([z * y, - z * y])
        rot_mat[0, 2, :] = np.array([x * y, - x * y])
        rot_mat[1, 0, :] = rot_mat[0, 1, :]
        rot_mat[1, 1, :] = np.array([z ** 2, 1.0 - z ** 2])
        rot_mat[1, 2, :] = np.array([z * x, - z * x])
        rot_mat[2, 0, :] = rot_mat[0, 2, :]
        rot_mat[2, 1, :] = rot_mat[1, 2, :]
        rot_mat[2, 2, :] = np.array([x ** 2, 1 - x ** 2])

        if self.rot_type == 'tensor':
            SKpp_rsp = th.reshape(SKpp, [2, 1])
            rot_mat_th = th.from_numpy(rot_mat).float()
            rot_mat_th.requires_grad = False
            hs = th.matmul(rot_mat_th, SKpp_rsp)
            hs = th.reshape(hs, [3, 3])
        else:
            SKpp_rsp = np.reshape(SKpp, [2, 1])
            hs = np.dot(rot_mat, SKpp_rsp)
            hs = np.reshape(hs, [3, 3])
        return hs

    def pd(self, Angvec, SKpd):
        x = Angvec[0]
        y = Angvec[1]
        z = Angvec[2]
        s3 = np.sqrt(3.0)
        rot_mat = np.zeros([5, 3, 2])
        rot_mat[0, 0, :] = np.array([-(-1.0 + z ** 2 + x ** 2) * x * s3,
                                     (2.0 * z ** 2 + 2.0 * x ** 2 - 1.0) * x])

        rot_mat[1, 0, :] = np.array([-(-1 + z ** 2 + x ** 2) * z * s3,
                                     (2 * z ** 2 + 2 * x ** 2 - 1.0) * z])

        rot_mat[2, 0, :] = np.array([y * (3 * z ** 2 - 1) / 2.0, -s3 * z ** 2 * y])

        rot_mat[3, 0, :] = np.array([s3 * y * x * z, -2 * x * y * z])

        rot_mat[4, 0, :] = np.array([y * s3 * (2 * x ** 2 - 1 + z ** 2) / 2.0,
                                     -(z ** 2 + 2 * x ** 2) * y])

        rot_mat[0, 1, :] = np.array([x * y * z * s3, -2.0 * z * x * y])

        rot_mat[1, 1, :] = np.array([y * (z ** 2) * s3, -(2.0 * z ** 2 - 1.0) * y])

        rot_mat[2, 1, :] = np.array([(z * (3.0 * z ** 2 - 1.0)) / 2.0,
                                     -z * s3 * (-1.0 + z ** 2)])

        rot_mat[3, 1, :] = np.array([x * z ** 2 * s3, -1 * (2.0 * z ** 2 - 1.0) * x])

        rot_mat[4, 1, :] = np.array([(2.0 * x ** 2 - 1.0 + z ** 2) * z * s3 / 2.0,
                                     -1 * (z * (2.0 * x ** 2 - 1.0 + z ** 2))])

        rot_mat[0, 2, :] = np.array([(x ** 2) * y * s3, -1 * (2.0 * x ** 2 - 1.0) * y])

        rot_mat[1, 2, :] = np.array([x * y * s3 * z, -2.0 * y * x * z])

        rot_mat[2, 2, :] = np.array([(x * (3.0 * z ** 2 - 1.0)) / 2.0,
                                     -1 * s3 * (z ** 2) * x])

        rot_mat[3, 2, :] = np.array([x ** 2 * s3 * z, -1 * (2.0 * x ** 2 - 1.0) * z])

        rot_mat[4, 2, :] = np.array([x * (2.0 * x ** 2 - 1.0 + z ** 2) * s3 / 2.0,
                                     -1 * ((z ** 2 - 2.0 + 2.0 * x ** 2) * x)])

        # [5,3,2] dot (2,1) => [5,3,1] => [5,3]
        if self.rot_type == 'tensor':
            SKpd_rsp = th.reshape(SKpd, [2, 1])
            rot_mat_th = th.from_numpy(rot_mat).float()
            rot_mat_th.requires_grad = False
            hs = th.matmul(rot_mat_th, SKpd_rsp)
            hs = th.reshape(hs, [5, 3])
        else:
            SKpd_rsp = np.reshape(SKpd, [2, 1])
            hs = np.dot(rot_mat, SKpd_rsp)
            hs = np.reshape(hs, [5, 3])
        return hs

    def dd(self, Angvec, SKdd):
        x = Angvec[0]
        y = Angvec[1]
        z = Angvec[2]
        s3 = np.sqrt(3.0)
        rot_mat = np.zeros([5, 5, 3])

        rot_mat[0, 0, :] = np.array([-3.0 * x ** 2 * (-1.0 + z ** 2 + x ** 2),
                                     (4.0 * x ** 2 * z ** 2 - z ** 2 + 4.0 * x ** 4 - 4.0 * x ** 2 + 1.0),
                                     (-x ** 2 * z ** 2 + z ** 2 + x ** 2 - x ** 4)])

        rot_mat[1, 0, :] = np.array([-3.0 * x * (-1.0 + z ** 2 + x ** 2) * z,
                                     (4.0 * z ** 2 + 4.0 * x ** 2 - 3.0) * z * x,
                                     -x * (z ** 2 + x ** 2) * z])

        rot_mat[2, 0, :] = np.array([x * y * s3 * (3.0 * z ** 2 - 1.0) / 2.0,
                                     -2.0 * s3 * y * x * (z ** 2),
                                     x * y * (z ** 2 + 1.0) * s3 / 2.0])

        rot_mat[3, 0, :] = np.array([3.0 * (x ** 2) * y * z,
                                     -(4.0 * x ** 2 - 1.0) * z * y,
                                     y * (-1.0 + x ** 2) * z])

        rot_mat[4, 0, :] = np.array([3.0 / 2.0 * y * x * (2.0 * x ** 2 - 1.0 + z ** 2),
                                     -2.0 * y * x * (2.0 * x ** 2 - 1.0 + z ** 2),
                                     y * x * (2.0 * x ** 2 - 1.0 + z ** 2) / 2.0])

        rot_mat[0, 1, :] = rot_mat[1, 0, :]

        rot_mat[1, 1, :] = np.array([-3.0 * (-1.0 + z ** 2 + x ** 2) * z ** 2,
                                     (4.0 * z ** 4 - 4.0 * z ** 2 + 4.0 * x ** 2 * z ** 2 + 1.0 - x ** 2),
                                     -(-1.0 + z) * (z ** 3 + z ** 2 + x ** 2 * z + x ** 2)])

        rot_mat[2, 1, :] = np.array([y * s3 * z * (3.0 * z ** 2 - 1.0) / 2.0,
                                     -z * s3 * (2.0 * z ** 2 - 1.0) * y,
                                     (-1.0 + z ** 2) * s3 * z * y / 2.0])

        rot_mat[3, 1, :] = np.array([3.0 * y * x * (z ** 2),
                                     -(4.0 * z ** 2 - 1.0) * y * x,
                                     x * y * (-1.0 + z ** 2)])

        rot_mat[4, 1, :] = np.array([3.0 / 2.0 * y * (2.0 * x ** 2 - 1.0 + z ** 2) * z,
                                     -(2.0 * z ** 2 - 1.0 + 4.0 * x ** 2) * z * y,
                                     y * (z ** 2 + 2.0 * x ** 2 + 1.0) * z / 2.0])

        rot_mat[0, 2, :] = rot_mat[2, 0, :]
        rot_mat[1, 2, :] = rot_mat[2, 1, :]

        rot_mat[2, 2, :] = np.array([((3.0 * z ** 2 - 1.0) ** 2) / 4.0,
                                     -(3.0 * (-1.0 + z ** 2) * z ** 2),
                                     3.0 / 4.0 * ((-1.0 + z ** 2) ** 2)])

        rot_mat[3, 2, :] = np.array([x * (3.0 * z ** 2 - 1.0) * s3 * z / 2.0,
                                     -(2.0 * z ** 2 - 1.0) * x * z * s3,
                                     z * x * s3 * (-1.0 + z ** 2) / 2.0])

        rot_mat[4, 2, :] = np.array([(2.0 * x ** 2 - 1.0 + z ** 2) * (3.0 * z ** 2 - 1.0) * s3 / 4.0,
                                     -(2.0 * x ** 2 - 1.0 + z ** 2) * (z ** 2) * s3,
                                     s3 * (2.0 * x ** 2 - 1.0 + z ** 2) * (z ** 2 + 1.0) / 4.0])

        rot_mat[0, 3, :] = rot_mat[3, 0, :]
        rot_mat[1, 3, :] = rot_mat[3, 1, :]
        rot_mat[2, 3, :] = rot_mat[3, 2, :]

        rot_mat[3, 3, :] = np.array([3.0 * x ** 2 * z ** 2,
                                     (-4.0 * x ** 2 * z ** 2 + z ** 2 + x ** 2),
                                     (-1.0 + z) * (-z + x ** 2 * z - 1.0 + x ** 2)])

        rot_mat[4, 3, :] = np.array([3.0 / 2.0 * x * (2.0 * x ** 2 - 1.0 + z ** 2) * z,
                                     -((2.0 * z ** 2 - 3.0 + 4.0 * x ** 2) * z * x),
                                     (x * (z ** 2 - 3.0 + 2.0 * x ** 2) * z) / 2.0])

        rot_mat[0, 4, :] = rot_mat[4, 0, :]
        rot_mat[1, 4, :] = rot_mat[4, 1, :]
        rot_mat[2, 4, :] = rot_mat[4, 2, :]
        rot_mat[3, 4, :] = rot_mat[4, 3, :]
        rot_mat[4, 4, :] = np.array([3.0 / 4.0 * ((2.0 * x ** 2 - 1.0 + z ** 2) ** 2),
                                     -z ** 4 + z ** 2 - 4.0 * x ** 2 * z ** 2 - 4.0 * x ** 4 + 4.0 * x ** 2,
                                     (z ** 4 / 4.0 + x ** 2 * z ** 2 + z ** 2 / 2.0 + 1.0 / 4.0 - x ** 2 + x ** 4)])

        # [5,5,3] dot (3,1) => [5,5,1] => [5,5]
        if self.rot_type == 'tensor':
            SKdd_rsp = th.reshape(SKdd, [3, 1])
            rot_mat_th = th.from_numpy(rot_mat).float()
            rot_mat_th.requires_grad = False
            hs = th.matmul(rot_mat_th, SKdd_rsp)
            hs = th.reshape(hs, [5, 5])
        else:
            SKdd_rsp = np.reshape(SKdd, [3, 1])
            hs = np.dot(rot_mat, SKdd_rsp)
            hs = np.reshape(hs, [5, 5])

        return hs

