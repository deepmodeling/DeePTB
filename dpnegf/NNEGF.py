import os
import time
import ase
import ase.io
import argparse
import numpy as np

from dpnegf.Parameters import Paras
from dpnegf.nnet.Model import Model
from dpnegf.negf.NEGFStruct import StructNEGFBuild
from dpnegf.negf.SurfaceGF import SurfGF
from dpnegf.negf.NEGFHamilton import NEGFHamiltonian, Device_Hamils, Contact_Hamils
from dpnegf.negf.NEGF import NEGFcal

from ase.transport.calculators import TransportCalculator

def deepnegf(args:argparse.Namespace):   
    """Perform NEGF simulations with NN-baed TB Hamiltonians.

    Args:
        args (argparse.Namespace): command line paras.

    Output:
        Transmission coefficient.
        Current.

    Return:
        None
    """
    time_start = time.time() 

    input_file = args.input_file
    fp = open(input_file)
    paras = Paras(fp,args.command, args.nn_off)
    if args.nn_off:
        paras.TBmodel = 'sktb'
    else:
        paras.TBmodel = 'nntb'

    structfile = args.struct
    structfmt = args.format

    structase = ase.io.read(structfile,format=structfmt)
    negfH = NEGFHamiltonian(paras,structase,conttol=1e-3)

    # calculate fermi level.
    natomcont = negfH.ngstr.GeoRegions['Source'][1]-negfH.ngstr.GeoRegions['Source'][0] + 1
    natomcontpl = natomcont//2
    soucepl1st = negfH.ngstr.GeoRegions['Source'][0]-1
    drainpl1st = negfH.ngstr.GeoRegions['Drain'][0]-1

    vecs_shift= (negfH.ngstr.GeoStruct.positions[soucepl1st:soucepl1st+natomcontpl] 
                - negfH.ngstr.GeoStruct.positions[drainpl1st+natomcontpl:drainpl1st+2*natomcontpl])

    assert negfH.ngstr.ExtDirect['Source'] == negfH.ngstr.ExtDirect['Drain']
    axistrans = negfH.ngstr.ExtDirect['Source']
    periodic_length = vecs_shift[0,axistrans]
    latticecell = np.zeros([3,3])
    for i in range(3):
        if i==axistrans:
            latticecell[i,i] = periodic_length
        else:
            latticecell[i,i] = 500

    _,scatterase = negfH.ngstr.BuildDevice(tag='Scatter')
    scatterase.cell = latticecell
    #scatterase.pbc = [False, False, False]
    scatterase.pbc[axistrans]=True
    
    mdl = Model(paras)
    mdl.structinput(scatterase)
    

    if args.nn_off:
        mdl.SKhoppings()
        mdl.HSmat(hoppings = mdl.skhoppings, overlaps = mdl.skoverlaps , 
              onsiteEs = mdl.onsiteEs, onsiteSs = mdl.onsiteSs)
    else:
        mdl.loadmodel()
        mdl.nnhoppings()
        mdl.SKhoppings()
        mdl.SKcorrection()
        mdl.HSmat(hoppings = mdl.hoppings_corr, overlaps = mdl.overlaps_corr ,  
              onsiteEs = mdl.onsiteEs_corr, onsiteSs = mdl.onsiteSs_corr)

    nk = paras.nkfermi
    klist = np.zeros([nk,3])
    klist[:,axistrans] = np.linspace(0,0.5,nk)[0:nk] 
    eigks = mdl.Eigenvalues(kpoints = klist)
    eigksnp =  eigks.detach().numpy()

    nk = eigksnp.shape[0]
    ValElec = np.asarray(mdl.bondbuild.ProjValElec)
    nume = np.sum(ValElec[mdl.bondbuild.TypeID])
    numek = nume * nk//paras.SpinDeg
    sorteigs =  np.sort(np.reshape(eigksnp,[-1]))
    EF=(sorteigs[numek] + sorteigs[numek-1])/2
    print('Efermi : %10.6f' %EF)

    time_measure = time.time() 
    print('Timing E-Fermi : %16.3f s' %(time_measure-time_start))

    ScatDict, ScatContDict = negfH.Scat_Hamils()
    ContDict = negfH.Cont_Hamils()

    negfcal = NEGFcal(paras)    
    
    NNEF = EF

    if not args.use_ase:
        paras.DeviceFermi = EF
        paras.ContactFermi = EF
        negfcal.Scat_Hamiltons(HamilDict = ScatDict,Efermi = paras.DeviceFermi)
        negfcal.Scat_Cont_Hamiltons(HamilDict = ScatContDict,Efermi = paras.DeviceFermi)
        negfcal.Cont_Hamiltons(HamilDict = ContDict,Efermi = paras.ContactFermi)
        time_measure = time.time() 
        print('Timing Region Hamiltonian : %16.3f s' %(time_measure-time_start))

        negfcal.get_current()
        np.save('transmission',{'E':negfcal.energies,'T':negfcal.transmission})
        np.save('current',{'bias':negfcal.bias,'current':negfcal.current})

        time_measure = time.time() 
        print('Timing current and transmission: %16.3f s' %(time_measure-time_start))
        print('Done!')

    else:
        print('Calculate negf use ase api.')
        negfcal.Scat_Hamiltons(HamilDict = ScatDict)
        negfcal.Scat_Cont_Hamiltons(HamilDict = ScatContDict)
        negfcal.Cont_Hamiltons(HamilDict = ContDict)
        
        h = ScatDict['Hss'][0]
        s = ScatDict['Sss'][0]
        norbs_h = ContDict['Source']['H00'][0].shape[0]

        h1 = np.zeros([norbs_h*2,norbs_h*2],dtype=complex)
        s1 = np.zeros([norbs_h*2,norbs_h*2],dtype=complex)

        h1[0:norbs_h,0:norbs_h] = ContDict['Source']['H00'][0]
        h1[0:norbs_h,norbs_h:2*norbs_h] = ContDict['Source']['H01'][0]
        h1[norbs_h:2*norbs_h,0:norbs_h] = ContDict['Source']['H01'][0].T.conj()
        h1[norbs_h:2*norbs_h,norbs_h:2*norbs_h] = ContDict['Source']['H00'][0]

        s1[0:norbs_h,0:norbs_h] = ContDict['Source']['S00'][0]
        s1[0:norbs_h,norbs_h:2*norbs_h] = ContDict['Source']['S01'][0]
        s1[norbs_h:2*norbs_h,0:norbs_h] = ContDict['Source']['S01'][0].T.conj()
        s1[norbs_h:2*norbs_h,norbs_h:2*norbs_h] = ContDict['Source']['S00'][0]

        norbs_h = ContDict['Drain']['H00'][0].shape[0]
        h2 = np.zeros([norbs_h*2,norbs_h*2],dtype=complex)
        s2 = np.zeros([norbs_h*2,norbs_h*2],dtype=complex)

        h2[0:norbs_h,0:norbs_h] = ContDict['Drain']['H00'][0]
        h2[0:norbs_h,norbs_h:2*norbs_h] =  ContDict['Drain']['H01'][0].T.conj()
        h2[norbs_h:2*norbs_h,0:norbs_h] =  ContDict['Drain']['H01'][0]
        h2[norbs_h:2*norbs_h,norbs_h:2*norbs_h] = ContDict['Drain']['H00'][0]

        s2[0:norbs_h,0:norbs_h] = ContDict['Drain']['S00'][0]
        s2[0:norbs_h,norbs_h:2*norbs_h] =  ContDict['Drain']['S01'][0].T.conj()
        s2[norbs_h:2*norbs_h,0:norbs_h] =  ContDict['Drain']['S01'][0]
        s2[norbs_h:2*norbs_h,norbs_h:2*norbs_h] = ContDict['Drain']['S00'][0]

        hc1 = ScatContDict['Source']['Hsc'][0].T.conj()
        sc1 = ScatContDict['Source']['Ssc'][0].T.conj()

        hc2 = ScatContDict['Drain']['Hsc'][0].T.conj()
        sc2 = ScatContDict['Drain']['Ssc'][0].T.conj()

        h -= NNEF*s
        h1 -= NNEF * s1
        h2 -= NNEF * s2
        hc1 -= NNEF * sc1
        hc2 -= NNEF * sc2
        time_measure = time.time() 
        print('Timing Region Hamiltonian : %16.3f s' %(time_measure-time_start))
        tcalc = TransportCalculator(h=h, h1=h2, h2=h1,  # hamiltonian matrices
                            s=s, s1=s2, s2=s1,  # overlap matrices
                            hc1=hc2,hc2=hc1, 
                            sc1=sc2,sc2=sc1,
                            eta=paras.eta,eta1=paras.eta,eta2=paras.eta,
                            dos=True)
        tcalc.set(energies=[0.0])
        G = tcalc.get_transmission()[0]
        print(f'Conductance: {G:.2f} 2e^2/h')

        tcalc.set(energies=np.linspace(paras.Emin,paras.Emax,paras.NumE))
        T = tcalc.get_transmission()
        np.save('Transition_apiase',{'E':tcalc.energies,'T':T})
        time_measure = time.time() 
        print('Timing Transmission : %16.3f s' %(time_measure-time_start))
        bias=np.linspace(paras.BiasV[0],paras.BiasV[1],paras.NumV)
        current = tcalc.get_current(bias)
        np.save('current_apiase',{'bias':bias,'current':current})
        time_measure = time.time() 
        print('Timing Current : %16.3f s' %(time_measure-time_start))
        print('Done!')


#if __name__ == "__main__":
#    deepnegf()