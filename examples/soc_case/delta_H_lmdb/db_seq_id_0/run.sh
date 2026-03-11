cp INPUT.scf INPUT
OMP_NUM_THREADS=1 mpirun -np 16 abacus | tee scf.log
cp INPUT.nscf INPUT
OMP_NUM_THREADS=1 mpirun -np 16 abacus | tee nscf.log
