from ase.io.trajectory import Trajectory
from ase.io.trajectory import TrajectoryWriter
import numpy as np
from ase.io import read, write

import re

#traj = Trajectory('./OUT.ABACUS/traindata/xdat.traj',mode='r')
atom = read('STRU')
writer = TrajectoryWriter("xdat.traj",mode='w', atoms=atom)
writer.write()

#kpoints = np.loadtxt('./OUT.ABACUS/kpoints')
band = np.loadtxt('./OUT.ABACUS_NSCF/BANDS_1.dat')
#np.save('kpoints',kpoints)
np.save('eigs',band[:,2+5:])

f=open('./OUT.ABACUS_NSCF/running_nscf.log','r')
datlines = f.readlines()
f.close()

kpt=[]
for i in range(len(datlines)):
    if re.findall('SETUP K-POINTS',datlines[i]):
        nspin = int(datlines[i+1].split('=')[-1])
        nkstot = int(datlines[i+2].split('=')[-1])
        for j in range(i+5, i+5+nkstot):
            line=datlines[j]
            kpt.append(np.array([float(ii) for ii in line.split()[1:-1]]))

        break
kpt = np.asarray(kpt)

np.save('kpoints',kpt)
