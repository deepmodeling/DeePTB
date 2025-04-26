#!/bin/bash 
source /opt/intel/oneapi/setvars.sh
export OMP_NUM_THREADS=1
cp ./scf/* ./ 
mpirun -np 16 abacus 
cp -r ./OUT.ABACUS ./OUT.ABACUS_SCF
cp ./nscf/* ./ 
mpirun -np 16 abacus 
mv  ./OUT.ABACUS ./OUT.ABACUS_NSCF 
cp ./scf/* ./ 
mv  ./OUT.ABACUS_SCF ./OUT.ABACUS 
rm  ./OUT.ABACUS_NSCF/*cube  
rm  ./OUT.ABACUS/*cube  
rm  ./*.orb  ./*.upf 
