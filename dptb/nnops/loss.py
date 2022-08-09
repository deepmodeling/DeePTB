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