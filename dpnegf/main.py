#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dpnegf.NNSKTB import deepTB, dpTBtrain, dpTBtest
from dpnegf.NNEGF import deepnegf

__author__ = "Q. Gu, et al."
__copyright__ = "Q. Gu & AISI"
__status__ = "developing"

def main():
    parser = argparse.ArgumentParser(
        description="DeepNEGF: A deep learning package for quantum transport simulations with"
        " non-equilibrium Green's function (NEGF) approach."
    ) 

    subparsers = parser.add_subparsers(title="valid subcommands", dest="command")

    paras_tbtrain = subparsers.add_parser("train", 
        help='train NN to get the TB hamiltonians with labeled data.')
    paras_tbtrain.add_argument("-r", "--restart", type=str, default = None,
        help='training the model initialed with checkpoint.')

    paras_test = subparsers.add_parser("test", 
        help='test the trained NN. using the unseen labeled data.')
    
    #paras_pred = subparsers.add_parser("predict", 
    #    help='using the trained NN to make prediction for unlabeled data.')
    
    paras_negf = subparsers.add_parser("negf", 
        help='Perform NEGF simulations with NN-baed TB Hamiltonians.')
    paras_negf.add_argument('-s', '--struct', type=str, default='struct.xyz',
        help='struct file name default struct.xyz')
    paras_negf.add_argument('-fmt', '--format', type=str, default='xyz',
        help='struct file format default xyz')

    args = parser.parse_args()

    if args.command=='train':
        dpTBtrain(args)
    elif args.command=='test':
        dpTBtest(args)
    elif args.command=='negf':
        deepnegf(args)
        
    
if __name__ == "__main__":
    main()


