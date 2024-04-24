from dptb.entrypoints.run import run
import pytest
import torch
import numpy as np


@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    """
    :return:
    """
    return str(request.config.rootdir)

# NEGF calculaion in 1D carbon chain with zero-bias transmission 1 G0

def test_negf_run_chain(root_directory):
    INPUT_file =  root_directory +"/dptb/tests/data/test_negf/test_negf_run/negf_chain_new.json" 
    output =  root_directory +"/dptb/tests/data/test_negf/test_negf_run/out_negf_chain"  
    checkfile =  root_directory +'/dptb/tests/data/test_negf/test_negf_run/nnsk_C_new.json'
    structure =  root_directory +"/dptb/tests/data/test_negf/test_negf_run/chain.vasp" 

    run(INPUT=INPUT_file,init_model=checkfile,structure=structure,output=output,\
    log_level=5,log_path=output+"/output.log")

    negf_results = torch.load(output+"/results/negf.out.pth")
    trans = negf_results['T_avg']
    assert(abs(trans[int(len(trans)/2)]-1)<1e-5)  #compare with calculated transmission at efermi



def test_negf_run_orth(root_directory):
    INPUT_file =  root_directory +"/dptb/tests/data/test_negf/test_negf_run/negf_graphene_new.json" 
    output =  root_directory +"/dptb/tests/data/test_negf/test_negf_run/out_negf_graphene"  
    checkfile =  root_directory +'/dptb/tests/data/test_negf/test_negf_run/nnsk_C_new.json'
    structure =  root_directory +"/dptb/tests/data/test_negf/test_negf_run/graphene.xyz" 

    run(INPUT=INPUT_file,init_model=checkfile,output=output,run_sk=True,structure=structure,\
    log_level=5,log_path=output+"/test.log",use_correction=False)

    negf_results = torch.load(output+"/results/negf.out.pth")

    k_standard = np.array([[0. , 0. , 0.], [0. , 0.33333333, 0.]])
    k = negf_results['k']
    assert(abs(k-k_standard).max()<1e-5)  #compare with calculated kpoints
      
    wk_standard = np.array([0.3333333333333333, 0.6666666666666666])
    wk = np.array(negf_results['wk'])
    assert abs(wk-wk_standard).max()<1e-5 #compare with calculated weight


    T_k0 = negf_results['T_k'][str(negf_results['k'][0])]
    T_k0_standard = [2.2307e-18, 7.0694e-18, 2.4631e-17, 9.6304e-17, 4.3490e-16, 2.3676e-15,
        1.6641e-14, 1.7068e-13, 3.3234e-12, 2.8054e-10, 9.9964e-01, 9.9985e-01,
        9.9989e-01, 9.9991e-01, 9.9991e-01, 9.9992e-01, 9.9992e-01, 9.9992e-01,
        9.9991e-01, 9.9987e-01, 4.0658e-08, 5.7304e-10, 8.4808e-11, 2.9762e-11,
        1.8432e-11, 1.8431e-11, 2.9762e-11, 8.4805e-11, 5.7300e-10, 4.0650e-08,
        9.9987e-01, 9.9991e-01, 9.9992e-01, 9.9992e-01, 9.9992e-01, 9.9991e-01,
        9.9991e-01, 9.9989e-01, 9.9985e-01, 9.9964e-01, 2.8058e-10, 3.3236e-12,
        1.7069e-13, 1.6642e-14, 2.3677e-15, 4.3491e-16, 9.6308e-17, 2.4632e-17,
        7.0696e-18, 2.2308e-18]
    T_k0_standard = torch.tensor(T_k0_standard)
    assert abs(T_k0-T_k0_standard).max()<1e-4

    T_k1 = negf_results['T_k'][str(negf_results['k'][1])]
    T_k1_standard = [3.4867e-19, 1.0166e-18, 3.2013e-18, 1.1041e-17, 4.2506e-17, 1.8749e-16,
        9.8430e-16, 6.5273e-15, 6.0546e-14, 9.6364e-13, 4.5495e-11, 3.3900e-07,
        9.9983e-01, 9.9988e-01, 9.9990e-01, 1.9996e+00, 1.9998e+00, 1.9998e+00,
        1.9998e+00, 1.9998e+00, 1.9998e+00, 9.9992e-01, 9.9992e-01, 9.9992e-01,
        9.9992e-01, 9.9992e-01, 9.9992e-01, 9.9992e-01, 9.9992e-01, 1.9998e+00,
        1.9998e+00, 1.9998e+00, 1.9998e+00, 1.9998e+00, 1.9996e+00, 9.9990e-01,
        9.9988e-01, 9.9983e-01, 3.3921e-07, 4.5502e-11, 9.6372e-13, 6.0549e-14,
        6.5277e-15, 9.8436e-16, 1.8749e-16, 4.2507e-17, 1.1042e-17, 3.2014e-18,
        1.0167e-18, 3.4868e-19]
    T_k1_standard = torch.tensor(T_k1_standard)
    assert  abs(T_k1-T_k1_standard).max()<1e-4


    T_avg = negf_results['T_avg']
    T_avg_standard = [9.7602e-19, 3.0342e-18, 1.0345e-17, 3.9462e-17, 1.7330e-16, 9.1420e-16,
        6.2031e-15, 6.1245e-14, 1.1482e-12, 9.4156e-11, 3.3321e-01, 3.3328e-01,
        9.9985e-01, 9.9989e-01, 9.9990e-01, 1.6664e+00, 1.6665e+00, 1.6665e+00,
        1.6665e+00, 1.6665e+00, 1.3332e+00, 6.6661e-01, 6.6661e-01, 6.6661e-01,
        6.6661e-01, 6.6661e-01, 6.6661e-01, 6.6661e-01, 6.6661e-01, 1.3332e+00,
        1.6665e+00, 1.6665e+00, 1.6665e+00, 1.6665e+00, 1.6664e+00, 9.9990e-01,
        9.9989e-01, 9.9985e-01, 3.3328e-01, 3.3321e-01, 9.4171e-11, 1.1482e-12,
        6.1249e-14, 6.2035e-15, 9.1425e-16, 1.7331e-16, 3.9464e-17, 1.0345e-17,
        3.0343e-18, 9.7605e-19]
    T_avg_standard = torch.tensor(T_avg_standard)
    assert  abs(T_avg-T_avg_standard).max()<1e-4  #compare with calculated transmission at efermi

def test_negf_run_S(root_directory):
    INPUT_file =  root_directory +"/dptb/tests/data/test_negf/test_negf_run/negf_graphene_new.json" 
    output =  root_directory +"/dptb/tests/data/test_negf/test_negf_run/out_negf_graphene"  
    checkfile =  root_directory +'/dptb/tests/data/test_negf/test_negf_run/nnsk_C_newS.json'
    structure =  root_directory +"/dptb/tests/data/test_negf/test_negf_run/graphene.xyz" 

    run(INPUT=INPUT_file,init_model=checkfile,output=output,run_sk=True,structure=structure,\
    log_level=5,log_path=output+"/test.log",use_correction=False)

    negf_results = torch.load(output+"/results/negf.out.pth")

    k_standard = np.array([[0. , 0. , 0.], [0. , 0.33333333, 0.]])
    k = negf_results['k']
    assert(abs(k-k_standard).max()<1e-5)  #compare with calculated kpoints
      
    wk_standard = np.array([0.3333333333333333, 0.6666666666666666])
    wk = np.array(negf_results['wk'])
    assert abs(wk-wk_standard).max()<1e-5 #compare with calculated weight


    T_k0 = negf_results['T_k'][str(negf_results['k'][0])]
    T_k0_standard = [5.3533e-16, 1.7583e-15, 6.6429e-15, 3.0157e-14, 1.7671e-13, 1.5268e-12,
        2.6431e-11, 2.9100e-09, 9.9979e-01, 9.9988e-01, 9.9991e-01, 9.9992e-01,
        9.9993e-01, 9.9993e-01, 9.9993e-01, 9.9993e-01, 9.9993e-01, 9.9993e-01,
        9.9991e-01, 9.9986e-01, 1.1692e-08, 3.8933e-10, 7.0993e-11, 2.7496e-11,
        1.8139e-11, 1.9213e-11, 3.3478e-11, 1.0907e-10, 1.0088e-09, 6.0637e-07,
        9.9988e-01, 9.9990e-01, 9.9991e-01, 9.9990e-01, 9.9990e-01, 9.9988e-01,
        9.9985e-01, 9.9969e-01, 3.1857e-10, 1.5012e-12, 4.5775e-14, 2.9746e-15,
        2.9804e-16, 3.9799e-17, 6.5382e-18, 1.2576e-18, 2.7403e-19, 6.6083e-20,
        1.7337e-20, 4.8842e-21]
    T_k0_standard = torch.tensor(T_k0_standard)
    assert abs(T_k0-T_k0_standard).max()<1e-4

    T_k1 = negf_results['T_k'][str(negf_results['k'][1])]
    T_k1_standard = [4.9559e-17, 1.3964e-16, 4.3406e-16, 1.5227e-15, 6.2301e-15, 3.1271e-14,
        2.0969e-13, 2.2174e-12, 5.6095e-11, 2.8496e-08, 9.9983e-01, 9.9989e-01,
        9.9991e-01, 9.9992e-01, 1.9996e+00, 1.9998e+00, 1.9998e+00, 1.9998e+00,
        1.9998e+00, 1.9998e+00, 1.9998e+00, 9.9992e-01, 9.9992e-01, 9.9992e-01,
        9.9992e-01, 9.9992e-01, 9.9992e-01, 9.9991e-01, 1.9995e+00, 1.9998e+00,
        1.9998e+00, 1.9998e+00, 1.9997e+00, 1.9996e+00, 9.9988e-01, 9.9984e-01,
        9.9968e-01, 2.5744e-10, 1.2387e-12, 3.7395e-14, 2.3958e-15, 2.3651e-16,
        3.1124e-17, 5.0410e-18, 9.5653e-19, 2.0574e-19, 4.9005e-20, 1.2706e-20,
        3.5398e-21, 1.0487e-21]
    T_k1_standard = torch.tensor(T_k1_standard)
    assert  abs(T_k1-T_k1_standard).max()<1e-4


    T_avg = negf_results['T_avg']
    T_avg_standard = [2.1148e-16, 6.7919e-16, 2.5037e-15, 1.1067e-14, 6.3055e-14, 5.2978e-13,
        8.9502e-12, 9.7148e-10, 3.3326e-01, 3.3329e-01, 9.9985e-01, 9.9990e-01,
        9.9991e-01, 9.9992e-01, 1.6664e+00, 1.6665e+00, 1.6665e+00, 1.6665e+00,
        1.6665e+00, 1.6665e+00, 1.3332e+00, 6.6661e-01, 6.6661e-01, 6.6661e-01,
        6.6661e-01, 6.6661e-01, 6.6661e-01, 6.6661e-01, 1.3330e+00, 1.3332e+00,
        1.6665e+00, 1.6665e+00, 1.6665e+00, 1.6664e+00, 9.9988e-01, 9.9985e-01,
        9.9974e-01, 3.3323e-01, 1.0702e-10, 5.2534e-13, 1.6856e-14, 1.1492e-15,
        1.2010e-16, 1.6627e-17, 2.8171e-18, 5.5635e-19, 1.2401e-19, 3.0499e-20,
        8.1390e-21, 2.3272e-21]
    T_avg_standard = torch.tensor(T_avg_standard)
    assert  abs(T_avg-T_avg_standard).max()<1e-4  #compare with calculated transmission at efermi