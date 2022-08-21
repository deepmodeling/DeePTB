import numpy as np
import torch as th
import torchsort

def loss_type1(criterion, eig_pred, eig_label,num_el,num_kp, band_min=0, band_max=None, spin_deg=2):
    norbs = eig_pred.shape[-1]
    nbanddft = eig_label.shape[-1]
    up_nband = min(norbs,nbanddft)
    num_val_band = int(num_el//spin_deg)
    num_k_val_band = int(num_kp * num_el // spin_deg)
    assert num_val_band <= up_nband
    if band_max is  None:
        band_max = up_nband
    else:
        assert band_max <= up_nband
    
    band_min = int(band_min)
    band_max = int(band_max)

    assert band_min < band_max
    # shape of eigs [batch_size, num_kp, num_bands]
    assert len(eig_pred.shape) == 3 and len(eig_label.shape) == 3

    # 对齐eig_pred和eig_label
    eig_pred_cut = eig_pred[:,:,band_min:band_max]
    eig_label_cut = eig_label[:,:,band_min:band_max]
    loss = criterion(eig_pred_cut,eig_label_cut)

    return loss

def loss_spectral(criterion, eig_pred, eig_label, emin, emax, num_omega=None, sigma=0.1, **kwargs):
    ''' use eigenvalues to calculate electronic spectral functions and the use the prediced and label spectral 
    function to calcualted loss . 
    '''
    # calculate spectral fucntion A(k,w):
    assert len(eig_pred.shape) == 3 and len(eig_label.shape) == 3
    if num_omega is None:
        num_omega = int((emax - emin)/sigma)
    omega = th.linspace(emin,emax,num_omega)
    spectral_lbl = cal_spectral_func(eigenvalues= eig_label, omega=omega, sigma=sigma)
    spectral_pred = cal_spectral_func(eigenvalues= eig_pred, omega=omega, sigma=sigma)
    loss = criterion(spectral_lbl, spectral_pred)
    
    return loss

def gauss(x,sig,mu=0):
    ## gaussion fucntion
    return th.exp(-(x-mu)**2/(2*sig**2)) * (1/(th.sqrt(2*np.pi)*sig))

def cal_spectral_func(eigenvalues,omega,sigma=0.1):
    nsnap, nkp, nband = eigenvalues.shape
    eigs_rsp = th.reshape(eigenvalues,[nsnap * nkp * nband,1])
    omega = th.reshape(omega,[1,-1])
    nomega = omega.shape[1]
    diffmax = omega - eigs_rsp
    gaussian_weight= gauss(diffmax,sigma)
    gaussian_weight_fmt = th.reshape(gaussian_weight,[nsnap, nkp, nband, nomega])
    # eigenvalues_fmt = np.reshape(eigenvalues,[nsnap, nkp, nband, 1])
    spectral_func = np.sum(gaussian_weight_fmt,axis=2)

    return spectral_func



