#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import ase 
import yaml
import matplotlib.pyplot as plt
import ase.visualize
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from dptb.Parameters import Paras
from dptb.nnet import Model
import pickle as pickle


def save_dict_to_yaml(dict_value: dict, save_path: str):
    """dict保存为yaml"""
    with open(save_path, 'w') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))

def read_yaml_to_dict(yaml_path: str, ):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value

def plot_band():
    input_file = input('input.json is ? default: input.json \n ::') or 'input.json'
    struct_file = input('struct file  is ? default: hBN.vasp\n ::') or 'hBN.vasp'
    #input_file='./input.json'
    fp = open(input_file)
    paras = Paras(fp)
    paras.istrain = False
    paras.ispredict = True
    paras.istest = False
    mdl = Model(paras)

    asestr=ase.io.read(struct_file)
    mdl.loadmodel()

    mdl.structinput(asestr)
    mdl.nnhoppings()
    mdl.SKhoppings()

    snapase = asestr
    lat = snapase.cell.get_bravais_lattice()
    print(lat.description())
    #lat.plot_bz(show=True)
    special_kp = lat.get_special_points()
    #spmap['M'] = np.array([0.5,0.5,1])
    #kpath=snapase.cell.bandpath('GMKG', npoints=120)
    path_point = input('KPATH: str, default: GMKG \n ::') or 'GMKG'
    num_kps = input('NKP: int, default: 120\n ::') or 120
    num_kps = int(num_kps)

    kpath=snapase.cell.bandpath(path_point, num_kps)
    xlist, high_sym_kpoints, labels = kpath.get_linear_kpoint_axis()
    klist = kpath.kpts

    mdl.SKcorrection()
    mdl.HSmat(hoppings = mdl.hoppings_corr, overlaps = mdl.overlaps_corr , 
                  onsiteEs = mdl.onsiteEs_corr, onsiteSs = mdl.onsiteSs_corr)

    eigks = mdl.Eigenvalues(kpoints = klist)
    eigksnp =  eigks.detach().numpy()

    nk = eigksnp.shape[0]
    ValElec = np.asarray(mdl.bondbuild.ProjValElec)
    nume = np.sum(ValElec[mdl.bondbuild.TypeID])
    numek = nume * nk//2
    sorteigs =  np.sort(np.reshape(eigksnp,[-1]))
    EF=(sorteigs[numek] + sorteigs[numek-1])/2

    # sktb 
    mdl.HSmat(hoppings = mdl.skhoppings, overlaps = mdl.skoverlaps , 
                  onsiteEs = mdl.onsiteEs, onsiteSs = mdl.onsiteSs)

    skeigks = mdl.Eigenvalues(kpoints = klist)
    skeigksnp =  skeigks.detach().numpy()

    nk = skeigksnp.shape[0]
    ValElec = np.asarray(mdl.bondbuild.ProjValElec)
    nume = np.sum(ValElec[mdl.bondbuild.TypeID])
    numek = nume * nk//2
    sorteigs =  np.sort(np.reshape(skeigksnp,[-1]))
    SKEF=(sorteigs[numek] + sorteigs[numek-1])/2

    # band_yaml = read_yaml_to_dict("band.yaml")
    # klist = np.asarray(band_yaml['klist'])
    # xcoords = np.asarray(band_yaml['xcoords'])
    # xlabel = np.asarray(band_yaml['label_xcoords'])
    # labels = (band_yaml['labels'])
    # eigenvalues = np.asarray(band_yaml['eigenvalues'])
    
    band = pickle.load(open('./band_structure.pickle', 'rb'))
    klist = np.asarray(band['klist'])
    xcoords = np.asarray(band['xcoords'])
    xlabel = np.asarray(band['label_xcoords'])
    labels = (band['labels'])
    eigenvalues = np.asarray(band['eigenvalues'])

    nk = len(klist)
    numek = nume * nk//2
    sorteigs =  np.sort(np.reshape(eigenvalues,[-1]))
    DFTEF=(sorteigs[numek] + sorteigs[numek-1])/2

    print('Finished calulating band structure, start ploting...')
    Emax = int(input('Emax? default: 15 \n ::') or 15)
    Emin = int(input('Emin? default: -22 \n ::') or -22)


    interp=1
    #if not GUI:
    plt.switch_backend('agg')
    plt.figure(figsize=(5,5),dpi=100)

    for ib in range(eigenvalues.shape[1]):
        plt.plot(xcoords[::interp],eigenvalues[::interp,ib]-DFTEF,'ko',ms=1)
    plt.plot(xcoords[::interp],eigenvalues[::interp,0] - DFTEF,'ko',ms=1, label='DFT')

    for i in range(skeigksnp.shape[1]):
        plt.plot(xlist, skeigksnp[:,i] - SKEF,'b-',lw=1,alpha=0.5)
    plt.plot(xlist, skeigksnp[:,0] - SKEF - 100,'b-',lw=1,alpha=0.5,label='sktb')

    for i in range(eigksnp.shape[1]):
        plt.plot(xlist, eigksnp[:,i] - EF,'r-',lw=1)
    plt.plot(xlist, eigksnp[:,0] - EF ,'r-',lw=1,label='dptb')


    for ii in high_sym_kpoints:
        plt.axvline(ii,color='gray',lw=0.5,ls='--')

    plt.axhline(0,ls='-.',lw=0.5, c='gray')

    plt.legend(loc=1,framealpha=1,fontsize=7)
    plt.tick_params(direction='in')
    plt.title('band structure plot',fontsize=8)
    plt.ylim(Emin,Emax)
    plt.xlim(xlist.min(),xlist.max())
    plt.ylabel('E - EF (eV)',fontsize=8)
    plt.yticks(fontsize=8)
    plt.xticks(high_sym_kpoints,labels,fontsize=8)
    plt.savefig('./band.png',dpi=300)
    print('saved band structure.')

if __name__ == "__main__":
    plot_band()