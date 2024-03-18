#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from ase.io import read
from ase.io.trajectory import Trajectory


def read2bin():
    parser = argparse.ArgumentParser(
        description="read2bin: read data to binary file format for deepTB code."
    )
    parser.add_argument("-in","--in_dir",type=str, default="./", help="dir path of data file, default is ./")
    parser.add_argument("-ef","--eig_file",type=str, default="eigenvalues.dat", help="eigenvalues file name, default is eigenvalues.dat")
    parser.add_argument("-kf","--kpoints_file",type=str, default="kpoints.dat",  help="kpoints filename, default is kpoints.dat")
    parser.add_argument("-sf","--struct_file",type=str,default="struct.vasp", help="structure file name, default is struct.vasp")
    args = parser.parse_args()

    fp = open(args.in_dir + '/' + args.eig_file,'r')
    for i in range(2):
        line = fp.readline()
    fp.close()
    nsp,nkp,nbnd = int(line.split()[1]), int(line.split()[2]),int(line.split()[3])
    data = np.loadtxt(args.in_dir + '/' + args.eig_file)
    eigvaules = np.reshape(data,[nsp,nkp,nbnd])

    kpoints = np.loadtxt(args.in_dir + '/' + args.kpoints_file)

    trajstrs = read(args.in_dir + '/' + args.struct_file, format='vasp',index=':')
    traj = Trajectory(args.in_dir + '/' + 'xdat.traj',mode='w')
    for i  in range(1):
        traj.write(atoms=trajstrs[i])  
    traj.close()

    np.save(args.in_dir + '/' + 'eigs.npy',eigvaules)
    np.save(args.in_dir + '/' + 'kpoints.npy',kpoints)


if __name__ == "__main__":
    read2bin()
