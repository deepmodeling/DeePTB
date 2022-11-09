import numpy as np
import torch as th
#import torchsort

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
    
    batch_size, num_kp, num_bands = eig_pred_cut.shape

    eig_pred_cut -= eig_pred_cut.reshape(batch_size,-1).min(dim=1)[0].reshape(batch_size,1,1)
    eig_label_cut -= eig_label_cut.reshape(batch_size,-1).min(dim=1)[0].reshape(batch_size,1,1)

    loss = criterion(eig_pred_cut,eig_label_cut)

    return loss

def loss_soft_sort(criterion, eig_pred, eig_label,num_el,num_kp, sort_strength=0.5, kmax=None, kmin=0, band_min=0, band_max=None, spin_deg=2, gap_penalty=False, fermi_band=0, eta=1e-2, **kwarg):
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
    
    if kmax is None:
        kmax = num_kp
    else:
        assert kmax <= num_kp
    
    band_min = int(band_min)
    band_max = int(band_max)

    assert band_min < band_max
    # shape of eigs [batch_size, num_kp, num_bands]
    assert len(eig_pred.shape) == 3 and len(eig_label.shape) == 3

    eig_pred_cut = eig_pred[:,kmin:kmax,band_min:band_max]
    eig_label_cut = eig_label[:,kmin:kmax,band_min:band_max]
    batch_size, num_kp, num_bands = eig_pred_cut.shape

    eig_pred_cut -= eig_pred_cut.reshape(batch_size,-1).min(dim=1)[0].reshape(batch_size,1,1)
    eig_label_cut -= eig_label_cut.reshape(batch_size,-1).min(dim=1)[0].reshape(batch_size,1,1)

    eig_pred_cut = th.reshape(eig_pred_cut, [-1,band_max-band_min])
    eig_label_cut = th.reshape(eig_label_cut, [-1,band_max-band_min])

    eig_pred_soft = torchsort.soft_sort(eig_pred_cut,regularization_strength=sort_strength)
    eig_label_soft = torchsort.soft_sort(eig_label_cut,regularization_strength=sort_strength)
 
    
    eig_pred_soft = th.reshape(eig_pred_soft, [batch_size, num_kp, num_bands])
    eig_label_soft = th.reshape(eig_label_soft, [batch_size, num_kp, num_bands])
    
    loss = criterion(eig_pred_soft,eig_label_soft)

    if gap_penalty:
        gap1 = eig_pred_soft[:,:,fermi_band+1] - eig_pred_soft[:,:,fermi_band]
        gap2 = eig_label_soft[:,:,fermi_band+1] - eig_label_soft[:,:,fermi_band]
        loss_gap = criterion(1.0/(gap1+eta), 1.0/(gap2+eta)) 

    if num_kp > 1:
        # randon choose nk_diff kps' eigenvalues to gen Delta eig.
        # nk_diff = max(nkps//4,1)     
        nk_diff = num_kp        
        k_diff_i = np.random.choice(num_kp,nk_diff,replace=False)
        k_diff_j = np.random.choice(num_kp,nk_diff,replace=False)
        while (k_diff_i==k_diff_j).all():
            k_diff_j = np.random.choice(num_kp, nk_diff, replace=False)
        eig_diff_lbl = eig_label_soft[:,k_diff_i,:] - eig_label_soft[:,k_diff_j,:]
        eig_ddiff_pred = eig_pred_soft[:,k_diff_i,:]  - eig_pred_soft[:,k_diff_j,:]
        loss_diff =  criterion(eig_diff_lbl, eig_ddiff_pred) 
        
        loss = (1*loss + 1*loss_diff)/2 
    
    if gap_penalty:
        loss = loss + 0.1*loss_gap 

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
    min1 = th.min(eig_label)
    min2 = th.min(eig_pred)
    min1.detach()
    eig_label = eig_label-min1.detach()
    eig_pred  = eig_pred - min2.detach()
    spectral_lbl = cal_spectral_func(eigenvalues= eig_label, omega=omega, sigma=sigma)
    spectral_pred = cal_spectral_func(eigenvalues= eig_pred, omega=omega, sigma=sigma)
    loss = criterion(spectral_lbl, spectral_pred)
    
    return loss

def gauss(x,sig,mu=0):
    ## gaussion fucntion
    #return th.exp(-(x-mu)**2/(2*sig**2)) * (1/((2*th.pi)**0.5*sig))
    return th.exp(-(x-mu)**2/(2*sig**2))


def cal_spectral_func(eigenvalues,omega,sigma=0.1):
    nsnap, nkp, nband = eigenvalues.shape
    eigs_rsp = th.reshape(eigenvalues,[nsnap * nkp * nband,1])
    omega = th.reshape(omega,[1,-1])
    nomega = omega.shape[1]
    diffmax = omega - eigs_rsp
    gaussian_weight= gauss(diffmax,sigma)
    gaussian_weight_fmt = th.reshape(gaussian_weight,[nsnap, nkp, nband, nomega])
    # eigenvalues_fmt = np.reshape(eigenvalues,[nsnap, nkp, nband, 1])
    spectral_func = th.sum(gaussian_weight_fmt,dim=2)

    return spectral_func



def loss_proj_env(criterion, eig_pred, eig_label, ev_pred, proj_label, band_min=0, band_max=None):
    # eig_pred [nsnap, nkp, n_band_tb], eig_label [nsnap, nkp, n_band_dft]
    # ev_pred [nsnap, nkp, n_band_tb, norb_tb], ev_label [nsnap, nkp, n_band_dft, nprojorb_dft]
    # orbmap_pred [{atomtype-orbtype:index}*nsnap], orbmap_label [{atomtype-orbtype:index}*nsnap]
    # fit_band ["N-0s","B-0s"] like this
    
    norbs = eig_pred.shape[-1]
    nbanddft = eig_label.shape[-1]
    up_nband = min(norbs,nbanddft)
    if band_max is  None:
        band_max = up_nband
    else:
        assert band_max <= up_nband
    
    band_min = int(band_min)
    band_max = int(band_max)

    nsnap, nkp, n_band_tb = eig_pred.shape
    wei = np.abs(ev_pred)**2
    wei_shp = wei[:,:,band_min:band_max,[0,3,1,2,5,8,6,7]]
    eig_pred_reshap = th.reshape(eig_pred[:,:,band_min:band_max], [nsnap,nkp, band_max - band_min,1])
    encoding_band_pred = th.sum(eig_pred_reshap * wei_shp,axis=2)

    eig_label_reshap = th.reshape(eig_label[:,:,band_min:band_max], [nsnap,nkp,band_max - band_min,1])
    wei_lbl_shp = proj_label[:,:,band_min:band_max]
    encoding_band_label = th.sum(eig_label_reshap * wei_lbl_shp,axis=2)
    
    loss = criterion(encoding_band_pred, encoding_band_label)

    return loss
