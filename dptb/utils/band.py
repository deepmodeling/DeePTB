import sys
import ase
import ase.io
import argparse
import numpy as np
import matplotlib.pyplot as plt

from dptb.Parameters import Paras
from dptb.nnet import Model


def  band_plot(args:argparse.Namespace):
    """
    Plot the band structure.

    Input: 
    ----- 

    """
    input_file = args.input_file
    fp = open(input_file)
    paras = Paras(fp, args.command, args.nn_off)
    mdl = Model(paras)
    mdl.loadmodel()

    if args.struct:
        strase = ase.io.read(args.struct,format=args.format)
    elif hasattr(paras,'struct') and hasattr(paras,'format'): 
        strase = ase.io.read(paras.struct,format=paras.format)
    else:
        print('please set the stucture and format tag.')
        sys.exit()

    mdl.structinput(strase)
    mdl.nnhoppings()
    mdl.SKhoppings()
    
    lat = strase.cell.get_bravais_lattice()
    #print(lat.description())
    #lat.plot_bz(show=True)
    special_kp = lat.get_special_points()
    #spmap['M'] = np.array([0.5,0.5,1])
    for ikey in paras.HighSymKps.keys():
        special_kp[ikey] = np.array(paras.HighSymKps[ikey])

    kpath=strase.cell.bandpath(paras.KPATH, npoints=paras.nkpoints)
    xlist, high_sym_kpoints, labels = kpath.get_linear_kpoint_axis()
    klist = kpath.kpts

    if args.nn_off:
        mdl.SKcorrection()
        mdl.HSmat(hoppings = mdl.hoppings_corr, overlaps = mdl.overlaps_corr , 
              onsiteEs = mdl.onsiteEs_corr, onsiteSs = mdl.onsiteSs_corr)
    else:
        mdl.HSmat(hoppings = mdl.skhoppings, overlaps = mdl.skoverlaps , 
              onsiteEs = mdl.onsiteEs, onsiteSs = mdl.onsiteSs)

    eigks = mdl.Eigenvalues(kpoints = klist)
    eigksnp =  eigks.detach().numpy()
    
    nk = eigksnp.shape[0]
    ValElec = np.asarray(mdl.bondbuild.ProjValElec)
    nume = np.sum(ValElec[mdl.bondbuild.TypeID])
    numek = nume * nk//paras.SpinDeg
    sorteigs =  np.sort(np.reshape(eigksnp,[-1]))
    EF=(sorteigs[numek] + sorteigs[numek-1])/2

    if paras.band_range != None:
        emax =  paras.band_range[1]
        emin =  paras.band_range[0]
    else:
        emin = np.min(eigksnp - EF)
        emax = np.max(eigksnp - EF)

    # uncomment in server without UI
    plt.switch_backend('agg')           
    plt.figure(figsize=(5,5),dpi=200)

    for i in range(eigksnp.shape[1]):
        plt.plot(xlist, eigksnp[:,i] - EF,'r-',lw=1)
    for ii in high_sym_kpoints:
        plt.axvline(ii,color='gray',lw=1,ls='--')

    plt.axhline(0,ls='-.',c='gray')
    #plt.legend(loc=1,framealpha=1)
    plt.tick_params(direction='in')
    plt.ylim(emin,emax)
    plt.xlim(xlist.min(),xlist.max())
    plt.ylabel('E - EF (eV)',fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(high_sym_kpoints,labels,fontsize=12)
    plt.savefig('./band.png',dpi=100)
    np.save('band_structure.npy',{'xlist':xlist,'eigenvalues':eigksnp,'Efermi':EF,'high_sym_kpoints':high_sym_kpoints,'labels':labels})

    
    plotcode=[["import numpy as np"],
    ["import matplotlib.pyplot as plt"],
    ["band_structure = np.load('band_structure.npy',allow_pickle=True).tolist()"],
    ["plt.switch_backend('agg')           "],
    ["plt.figure(figsize=(5,5),dpi=200)"],
    ["for i in range(band_structure['eigenvalues'].shape[1]):"],
    ["    plt.plot(band_structure['xlist'], band_structure['eigenvalues'][:,i] - band_structure['Efermi'],'r-',lw=1)"],
    ["for ii in band_structure['high_sym_kpoints']:"],
    ["    plt.axvline(ii,color='gray',lw=1,ls='--')"],
    ["plt.axhline(0,ls='-.',c='gray')"],
    ["#plt.legend(loc=1,framealpha=1)"],
    ["plt.tick_params(direction='in')"],
    ["#plt.ylim(emin,emax)"],
    ["plt.xlim(band_structure['xlist'].min(),band_structure['xlist'].max())"],
    ["plt.ylabel('E - EF (eV)',fontsize=12)"],
    ["plt.yticks(fontsize=12)"],
    ["plt.xticks(band_structure['high_sym_kpoints'],band_structure['labels'],fontsize=12)"],
    ["plt.savefig('./band_plot.png',dpi=100)"]]
    
    f=open('showband.py','w')
    for iraw in plotcode:
        print(iraw[0],file=f)
 