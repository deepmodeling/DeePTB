# contents of test_append.py
import pytest
import numpy as np
import torch as th
from dptb.hamiltonian.transform_sk import RotationSK

class TestRotationSK:

    rotmap = RotationSK(rot_type=th.float32, device='cpu')
    rotmap_th = RotationSK(rot_type=th.float32, device='cpu')


    def test_ss(self):
        vec = np.random.uniform(size=[3])
        vec = vec/np.linalg.norm(vec)
        Hvalue = np.random.uniform(size=[1,1])
        
        
        Hvalue = th.from_numpy(Hvalue).float()
        res = self.rotmap_th.rot_HS('ss', Hvalue=Hvalue, Angvec=th.tensor(vec))
        assert (np.abs(res.numpy() - Hvalue.numpy()) < 1e-6)


    def test_sp(self):
        vec = np.random.uniform(size=[3])
        vec = vec/np.linalg.norm(vec)
        Hvalue = np.random.uniform(size=[1,1])
        
        rot_mat = np.reshape(vec[[1,2,0]] ,[3 ,1])
        spvalue = np.reshape(Hvalue ,[1 ,1])
        hs = np.reshape(np.dot(rot_mat ,spvalue),[3 ,1])

        Hvalue = th.from_numpy(Hvalue).float()
        res = self.rotmap_th.rot_HS('sp', Hvalue=Hvalue, Angvec=th.tensor(vec))
        hs = np.asarray(hs,dtype=np.float32)
        assert (np.abs(res.numpy() - hs) < 1e-6).all()

 
    def test_sd(self):
        vec = np.random.uniform(size=[3])
        vec = vec/np.linalg.norm(vec)
        Hvalue = np.random.uniform(size=[1,1])

        x,y,z = vec
        s3 = np.sqrt(3.0)
        rot_mat = np.array([s3 * x *y, s3 * y *z, 1.5 * z**2 -0.5, \
                            s3 * x * z, s3 * (2.0 * x ** 2 - 1.0 + z ** 2) / 2.0])
        rot_mat = np.reshape(rot_mat, [5, 1])
        hs = np.reshape(np.dot(rot_mat,np.reshape(Hvalue,[1,1])),[5,1])

        Hvalue = th.from_numpy(Hvalue).float()
        res = self.rotmap_th.rot_HS('sd', Hvalue=Hvalue, Angvec=th.tensor(vec))
        hs = np.asarray(hs,dtype=np.float32)
        assert (np.abs(res.numpy() - hs) < 1e-6).all()




    def test_pp(self):
        vec = np.random.uniform(size=[3])
        vec = vec/np.linalg.norm(vec)
        Hvalue = np.random.uniform(size=[2,1])
        
        x,y,z = vec
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

        hs = np.reshape(np.dot(rot_mat,np.reshape(Hvalue,[2,1])),[3,3])

        Hvalue = th.from_numpy(Hvalue).float()
        res = self.rotmap_th.rot_HS('pp', Hvalue=Hvalue, Angvec=th.tensor(vec))
        hs = np.asarray(hs,dtype=np.float32)
        assert (np.abs(res.numpy() - hs) < 1e-6).all()


    def test_pd(self):
        vec = np.random.uniform(size=[3])
        vec = vec/np.linalg.norm(vec)
        Hvalue = np.random.uniform(size=[2,1])
        
        x,y,z = vec
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

        hs = np.reshape(np.dot(rot_mat,np.reshape(Hvalue,[2,1])),[5,3])

        Hvalue = th.from_numpy(Hvalue).float()
        res = self.rotmap_th.rot_HS('pd', Hvalue=Hvalue, Angvec=th.tensor(vec))
        hs = np.asarray(hs,dtype=np.float32)
        assert (np.abs(res.numpy() - hs) < 1e-6).all()


    def test_dd(self):
        vec = np.random.uniform(size=[3])
        vec = vec/np.linalg.norm(vec)
        Hvalue = np.random.uniform(size=[3,1])
        
        x,y,z = vec
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

        hs = np.reshape(np.dot(rot_mat,np.reshape(Hvalue,[3,1])),[5,5])

        Hvalue = th.from_numpy(Hvalue).float()
        res = self.rotmap_th.rot_HS('dd', Hvalue=Hvalue, Angvec=th.tensor(vec))
        hs = np.asarray(hs,dtype=np.float32)
        assert (np.abs(res.numpy() - hs) < 1e-6).all()