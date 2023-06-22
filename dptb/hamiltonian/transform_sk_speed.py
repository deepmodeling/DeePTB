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
        epAngvec = th.tensor([0.3, 0.4, 0.5]) * (2**0.5)
        ep1 = th.tensor(-2.7)
        ep2 = th.tensor([-2.7, -3.1])
        ep3 = th.tensor([-2.7, -3.1, -3.5])
        self.ss = th.jit.trace(ss, [epAngvec, ep1])
        self.sp = th.jit.trace(sp, [epAngvec, ep1])
        self.sd = th.jit.trace(sd, [epAngvec, ep1])
        self.pp = th.jit.trace(pp, [epAngvec, ep2])
        self.pd = th.jit.trace(pd, [epAngvec, ep2])
        self.dd = th.jit.trace(dd, [epAngvec, ep3])

        # self.sd = sd
        # self.pd = pd
        # self.dd = dd

    def rot_HS(self, Htype, Hvalue, Angvec):
        assert Htype in h_all_types, "Wrong hktypes"

        switch = {'ss': self.ss,
                  'sp': self.sp,
                  'sd': self.sd,
                  'pp': self.pp,
                  'pd': self.pd,
                  'dd': self.dd}

        hs = switch.get(Htype)(Angvec, Hvalue).type(self.rot_type)
        hs.to(self.device)

        return hs


def ss(Angvec: th.Tensor, SKss: th.Tensor):
    ## ss orbital no angular dependent.
    return SKss.reshape(1,1)

def sp(Angvec: th.Tensor, SKsp: th.Tensor):
    #rot_mat = Angvec[[1,2,0]].reshape([3 ,1])
    hs = Angvec.view(-1) * SKsp.view(-1)
    return hs[[1,2,0]].reshape(3,1)

def sd(Angvec: th.Tensor, SKsd: th.Tensor):
    x = Angvec[0]
    y = Angvec[1]
    z = Angvec[2]

    s3 = 3**0.5

    rot_mat = th.stack([s3 * x * y, s3 * y *z, 1.5 * z**2 -0.5, \
                        s3 * x * z, s3 * (2.0 * x ** 2 - 1.0 + z ** 2) / 2.0])
    # [5,1]*[1,1] => [5,1]
    hs = rot_mat * SKsd.view(-1)

    return hs.reshape(5,1)

def pp(Angvec: th.Tensor, SKpp: th.Tensor):

    Angvec = Angvec[[1,2,0]].reshape(3,1)
    mat = th.matmul(Angvec, Angvec.T)
    rot_mat = th.stack([mat, th.eye(3)-mat], dim=-1)

    hs = th.matmul(rot_mat, SKpp.view(2,1))

    return hs.squeeze(-1)

def pd(Angvec: th.Tensor, SKpd: th.Tensor):
    p = Angvec[[1,2,0]].reshape(1,3)
    x,y,z = Angvec[0], Angvec[1], Angvec[2]
    s3 = 3**0.5
    d = th.stack([s3*x*y, s3*y*z, 0.5*(3*z*z-1), s3*x*z, 0.5*s3*(x*x-y*y)]).reshape(5,1)
    pd = th.matmul(d,p)
    fm = th.stack([x,0*x,y,z,y,0*x,-s3/3*y,2*s3/3*z,-s3/3*x,0*x,x,z,-y,0*x,x]).reshape(5,3)
    rot_mat = th.stack([pd, fm-2*s3/3*pd], dim=-1)

    hs = th.matmul(rot_mat, SKpd)
    hs = th.reshape(hs, [5, 3])

    return hs

def dd(Angvec, SKdd):

    x,y,z = Angvec[0], Angvec[1], Angvec[2]
    x2, y2, z2 = x**2, y**2, z**2
    xy, yz, zx = x*y, y*z, z*x
    s3 = 3**0.5
    d = th.stack([s3*xy, s3*yz, 0.5*(3*z2-1), s3*zx, 0.5*s3*(x2-y2)]).reshape(5,1)
    dd0 = d @ d.T
    dd2 = 1/3 * dd0
    dd2 = dd2 + th.stack([
        z2,-zx,2/s3*xy,-yz,0*x, \
        -zx,x2, -s3/3*yz,-xy,yz, \
        2/s3*xy,-s3/3*yz,2/3-z2,-s3/3*zx,s3/3*(x2-y2), \
        -yz,-xy,-s3/3*zx,y2,-zx, \
        0*x,yz,s3/3*(x2-y2),-zx,z2
    ]).reshape(5,5)
    rot_mat = th.stack([dd0, th.eye(5)-dd0-dd2, dd2], dim=-1)

    hs = th.matmul(rot_mat, SKdd.view(-1))

    return hs
