#!/usr/bin/env python3
import ase
import sys
import argparse
import ase.io as io
import numpy as np
import ase.neighborlist



def main():    
    parser = argparse.ArgumentParser(description="get the neighbors distance for a given struct") 
    parser.add_argument("struct", help="the struct file", type=str, default=None)
    parser.add_argument("-fmt","--fromat", type=str,default=None,help="the format string of the struct.")
    args = parser.parse_args()

    if args.struct is None:
        print('Error, you must specify struct! check help use -h')
        sys.exit(1)
    elif args.fromat is not None:
        structase = io.read(args.struct,format=args.fromat)
    else:
        print()
        print('Warning, you did not specify struct format, will detect the format by suffix.')
        print()
        structase = io.read(args.struct)
    

    ilist, jlist, Rlatt = ase.neighborlist.neighbor_list(quantities=['i', 'j', 'S'], a=structase, cutoff=10)
    shift_vec = structase.positions[jlist] - structase.positions[ilist] + np.matmul(Rlatt, np.array(structase.cell))
    norm = np.linalg.norm(shift_vec, axis=1)
    norm = np.reshape(norm, [-1, 1])

    dist_floor = np.sort(np.unique(np.floor(norm*100)/100))

    dist_ceil = np.sort(np.unique(np.ceil(norm*100)/100))
    print('<>'*42)
    print('<>'*15 + '   The floor distance:  ' + '<>'*15)
    ic =0
    for i in dist_floor[:]:
        print('%8.2f' %i, end='')
        ic+=1
        if ic%10==0:
            print()
    if len(dist_floor)%10!=0:
        print()
    print('<>'*42)

    print('<>'*15 + '  The ceiling distance: ' + '<>'*15)

    ic =0
    for i in dist_ceil[:]:
        print('%8.2f' %i, end='')
        ic+=1
        if ic%10==0:
            print()
    if len(dist_ceil)%10!=0:
        print()
    print('<>'*42)

if __name__ == '__main__':
    main()