import torch as th
import numpy as np

class lossfunction(object):
    def __init__(self,criterion):
        self.criterion =criterion

    def eigs_l2(self, eig_pred, eig_label, band_min=0, band_max=None, emax=None, emin=None, spin_deg=2, **kwargs):
        norbs = eig_pred.shape[-1]
        nbanddft = eig_label.shape[-1]
        num_kp = eig_label.shape[-2]
        assert num_kp == eig_pred.shape[-2]
        up_nband = min(norbs,nbanddft)

        if band_max is  None:
            band_max = up_nband
        else:
            assert band_max <= up_nband

        band_min = int(band_min)
        band_max = int(band_max)

        assert band_min < band_max
        assert len(eig_pred.shape) == 3 and len(eig_label.shape) == 3

        # 对齐eig_pred和eig_label
        eig_pred_cut = eig_pred[:,:,band_min:band_max]
        eig_label_cut = eig_label[:,:,band_min:band_max]


        batch_size, num_kp, num_bands = eig_pred_cut.shape

        eig_pred_cut = eig_pred_cut - eig_pred_cut.reshape(batch_size,-1).min(dim=1)[0].reshape(batch_size,1,1)
        eig_label_cut = eig_label_cut - eig_label_cut.reshape(batch_size,-1).min(dim=1)[0].reshape(batch_size,1,1)

        
        if emax != None and emin != None:
            mask_in = eig_label_cut.lt(emax) * eig_label_cut.gt(emin)
        elif emax != None:
            mask_in = eig_label_cut.lt(emax)
        elif emin != None:
            mask_in = eig_label_cut.gt(emin)
        else:
            mask_in = None

        loss = 0.0
        if mask_in is not None:
            if th.any(mask_in).item():
                loss = self.criterion(eig_pred_cut.masked_select(mask_in), eig_label_cut.masked_select(mask_in))
        else:
            loss = self.criterion(eig_pred_cut, eig_label_cut)

        return loss

    def eigs_l2dsf(self, eig_pred, eig_label, kmax=None, kmin=0, band_min=0, band_max=None, emax=None, emin=None, 
                   spin_deg=2, gap_penalty=False, fermi_band=0, eta=1e-2, eout_weight=0, nkratio=None, weight=1., **kwarg):
        norbs = eig_pred.shape[-1]
        nbanddft = eig_label.shape[-1]
        num_kp = eig_label.shape[-2]
        assert num_kp == eig_pred.shape[-2]
        up_nband = min(norbs,nbanddft)
        
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

        if isinstance(weight, list):
            assert len(weight) < up_nband, "band weight is overlength!"
        elif isinstance(weight, float) or isinstance(weight, int):
            if abs(weight - 1) < 1e-6:
                weight = 1.
            else:
                weight = [weight]*(band_max-band_min)
        else:
            raise TypeError

        assert band_min < band_max
        # shape of eigs [batch_size, num_kp, num_bands]
        assert len(eig_pred.shape) == 3 and len(eig_label.shape) == 3

        eig_pred_cut = eig_pred[:,kmin:kmax,band_min:band_max]
        eig_label_cut = eig_label[:,kmin:kmax,band_min:band_max]
        batch_size, num_kp, num_bands = eig_pred_cut.shape

        if isinstance(weight, list):
            if len(weight) < num_bands:
                weight = weight + (num_bands - len(weight))*[1.]
            weight = th.tensor(weight).reshape(1,1,-1)
        

        eig_pred_cut = eig_pred_cut - eig_pred_cut.reshape(batch_size,-1).min(dim=1)[0].reshape(batch_size,1,1)
        eig_label_cut = eig_label_cut - eig_label_cut.reshape(batch_size,-1).min(dim=1)[0].reshape(batch_size,1,1)

        if nkratio is not None:
            assert nkratio > 0.0
            assert nkratio <= 1.0
            random_mask = th.rand_like(eig_pred_cut) > nkratio    
            eig_pred_cut = eig_pred_cut.masked_fill(random_mask, 0.0)
            eig_label_cut = eig_label_cut.masked_fill(random_mask, 0.0)

        if emax != None and emin != None:
            mask_in = eig_label_cut.lt(emax) * eig_label_cut.gt(emin)
            mask_out = eig_label_cut.gt(emax) + eig_label_cut.lt(emin)
        elif emax != None:
            mask_in = eig_label_cut.lt(emax)
            mask_out = eig_label_cut.gt(emax)
        elif emin != None:
            mask_in = eig_label_cut.gt(emin)
            mask_out = eig_label_cut.lt(emin)
        else:
            mask_in = None
            mask_out = None

        # eig_pred_cut = th.reshape(eig_pred_cut, [-1,band_max-band_min])
        # eig_label_cut = th.reshape(eig_label_cut, [-1,band_max-band_min])

        # eig_pred_soft = torchsort.soft_sort(eig_pred_cut,regularization_strength=strength)
        # eig_label_soft = torchsort.soft_sort(eig_label_cut,regularization_strength=strength)
       
        # eig_pred_soft = th.reshape(eig_pred_soft, [batch_size, num_kp, num_bands])
        # eig_label_soft = th.reshape(eig_label_soft, [batch_size, num_kp, num_bands])        

        if not isinstance(weight, float):
            eig_pred_cut = eig_pred_cut * weight
            eig_label_cut = eig_label_cut * weight

        loss = 0
        if mask_in is not None:
            if th.any(mask_in).item():
                loss = loss + self.criterion(eig_pred_cut.masked_select(mask_in), eig_label_cut.masked_select(mask_in))
            if th.any(mask_out).item():
                loss = loss + eout_weight * self.criterion(eig_pred_cut.masked_select(mask_out), eig_label_cut.masked_select(mask_out))
        else:
            loss = self.criterion(eig_pred_cut, eig_label_cut)

        #print(loss)

        if gap_penalty:
            gap1 = eig_pred_cut[:,:,fermi_band+1] - eig_pred_cut[:,:,fermi_band]
            gap2 = eig_label_cut[:,:,fermi_band+1] - eig_label_cut[:,:,fermi_band]
            loss_gap = self.criterion(1.0/(gap1+eta), 1.0/(gap2+eta))

        if num_kp > 1:
            # randon choose nk_diff kps' eigenvalues to gen Delta eig.
            # nk_diff = max(nkps//4,1)     
            nk_diff = num_kp
            k_diff_i = np.random.choice(num_kp,nk_diff,replace=False)
            k_diff_j = np.random.choice(num_kp,nk_diff,replace=False)
            while (k_diff_i==k_diff_j).all():
                k_diff_j = np.random.choice(num_kp, nk_diff, replace=False)

            if mask_in is not None:
                eig_diff_lbl = eig_label_cut.masked_fill(mask_in, 0.)[:, k_diff_i,:] - eig_label_cut.masked_fill(mask_in, 0.)[:,k_diff_j,:]
                eig_ddiff_pred = eig_pred_cut.masked_fill(mask_in, 0.)[:,k_diff_i,:] - eig_pred_cut.masked_fill(mask_in, 0.)[:,k_diff_j,:]
            else:
                eig_diff_lbl = eig_label_cut[:,k_diff_i,:] - eig_label_cut[:,k_diff_j,:]
                eig_ddiff_pred = eig_pred_cut[:,k_diff_i,:]  - eig_pred_cut[:,k_diff_j,:]
            loss_diff =  self.criterion(eig_diff_lbl, eig_ddiff_pred) 

            loss = (1*loss + 1*loss_diff)/2
        
        if gap_penalty:
            loss = loss + 0.1*loss_gap 

        return loss


    def loss_spectral(self, eig_pred, eig_label, emin, emax, num_omega=None, sigma=0.1, **kwargs):
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
        spectral_lbl = self.cal_spectral_func(eigenvalues= eig_label, omega=omega, sigma=sigma)
        spectral_pred = self.cal_spectral_func(eigenvalues= eig_pred, omega=omega, sigma=sigma)
        loss = self.criterion(spectral_lbl, spectral_pred)

        return loss

    def gauss(self, x,sig,mu=0):
        ## gaussion fucntion
        #return th.exp(-(x-mu)**2/(2*sig**2)) * (1/((2*th.pi)**0.5*sig))
        return th.exp(-(x-mu)**2/(2*sig**2))


    def cal_spectral_func(self, eigenvalues,omega,sigma=0.1):
        nsnap, nkp, nband = eigenvalues.shape
        eigs_rsp = th.reshape(eigenvalues,[nsnap * nkp * nband,1])
        omega = th.reshape(omega,[1,-1])
        nomega = omega.shape[1]
        diffmax = omega - eigs_rsp
        gaussian_weight= self.gauss(diffmax,sigma)
        gaussian_weight_fmt = th.reshape(gaussian_weight,[nsnap, nkp, nband, nomega])
        # eigenvalues_fmt = np.reshape(eigenvalues,[nsnap, nkp, nband, 1])
        spectral_func = th.sum(gaussian_weight_fmt,dim=2)

        return spectral_func



    def loss_proj_env(self, eig_pred, eig_label, ev_pred, proj_label, band_min=0, band_max=None):
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

        loss = self.criterion(encoding_band_pred, encoding_band_label)

        return loss

    def block_l2(self, pred, label, **kwargs):
        """Hamiltonian block as training labels, such as wannier TB hamiltonians. The loss the is defined as deviation function of two list
        of tensors.

        Args:
            pred (list(torch.Tensor)): predicted hamiltonian blocks
            label (list(torch.Tensor)): labeled hamiltonian blocks e.g. Wannier TB hamiltonians
        """
        
        assert len(pred) == len(label)
        loss = 0
        count = 1
        for st in range(len(pred)):
            for p, l in zip(pred[st], label[st]):
                rd = np.random.randint(low=0, high=10)
                # only calculate loss for 70% of the blocks
                #if rd >= 3:
                loss += self.criterion(l, p)
                count += 1
        
        return loss / count
