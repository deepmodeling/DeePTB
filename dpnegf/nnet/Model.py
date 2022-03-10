import numpy as np
import torch  as th
import warnings
import torchsort

from dpnegf.sktb.StructData import BondListBuild
from dpnegf.sktb.SlaterKosterPara import SlaterKosterInt
from dpnegf.sktb.RotateSK import RotateHS
from dpnegf.nnet.Data import EnvBuild, DataLoad
from dpnegf.nnet.NetWorks import BuildNN,Index_Mapings

#from softsort.pytorch_ops import soft_sort

class Model(object):
    """ The nerual network model. can be used to train, test and predict. 
    
    Attributes:
        class:
            paras : Paras class.
            envbuild: EnvBuild class.
            bondbuild: BondListBuild class.
            dataload: DataLoad class.
            SKIntIns: SlaterKosterInt class.
            buildNN: BuildNN class.
        and other parameters for neural network model:
        SKAnglrMHSID: the index of orbit-orbit hoppings in the sk files. see sk file format.
        au2Ang: atomic unit to angstrom.
        istrain: true or false; istest: true or false; ispredict: true or false; 
            control the task is to train the NN model or test or use the NN model as a predictor.
        AtomType: atom types in the full struct. ProjAtomType: atom  types to be projected. ProjAnglrM: the angular momentum to project.
            the AtomType and ProjAtomType corresponds to different NN works for different types atoms of the bond and local environment.
        active_func: activation function.
        env_neurons: hyper paras of neural network for encoding the local environments as input.
        bond_neurons: hyper paras of neural network for hoppings prediction.
        onsite_neurons: hyper paras of neural network for onsite energies prediction.
    """
    def __init__(self,paras):
        # S-K files parameters
        self.SKAnglrMHSID = {'dd':np.array([0,1,2]), 
                     'dp':np.array([3,4]), 'pd':np.array([3,4]), 
                     'pp':np.array([5,6]), 
                     'ds':np.array([7]),   'sd':np.array([7]),
                     'ps':np.array([8]),   'sp':np.array([8]),
                     'ss':np.array([9])}
        self.au2Ang = 0.529177249 
        self.envbuild = EnvBuild(paras)
        self.bondbuild = BondListBuild(paras) 
        self.dataload = DataLoad(paras)
        
        self.SKIntIns = SlaterKosterInt(paras)
        self.SKIntIns.ReadSKfiles()
        self.SKIntIns.IntpSKfunc()
        self.NumHvals = self.SKIntIns.NumHvals
        
        self.rotHS = RotateHS(rot_type='tensor')

        self.paras   = paras
        self.istrain = paras.istrain
        self.istest  = paras.istest
        self.ispredict = paras.ispredict  
        self.correction_mode = int(paras.correction_mode)
        if self.correction_mode != 1 and self.correction_mode != 2:
            self.correction_mode = 1
            warnings.warn("correction mode setting error. I set it to default 1")
        
        self.SpinDeg = paras.SpinDeg
        self.AtomType  = self.envbuild.AtomType        # atom types in the full struct.
        self.ProjAtomType = self.envbuild.ProjAtomType      # atom  types to be projected.
        self.ProjAnglrM = self.envbuild.ProjAnglrM      # the angular momentum to project.
        self.AnglrMID = self.bondbuild.AnglrMID 

        if self.istrain or self.istest or self.ispredict:
            self.use_E_win = paras.use_E_win
            self.use_I_win = paras.use_I_win
            self.band_min = paras.band_min

            if self.use_E_win:
                self.energy_max = paras.energy_max
                # self.use_E_win
            elif self.use_I_win:
                # band index window
                self.band_max = paras.band_max
                # turn off the tag for reference struct training.
                self.withref = False
            else:
                print('error in define a window for trainng eigenvalues ')
                
        # self.band_window = paras.band_window
        # self.val_window = paras.val_window
        # assert (self.band_window[0] == self.val_window[0] and self.band_window[0] >= self.val_window[0])
        # self.cond_window = [self.val_window[1], self.band_window[1]]
        # self.val_cond_ratio = paras.val_cond_ratio


        # define and build neural networks.
        self.buildNN  = BuildNN(envtype= self.AtomType, bondtype = self.ProjAtomType, proj_anglr_m = self.ProjAnglrM) 
        self.IndMap = Index_Mapings(envtype= self.AtomType, bondtype = self.ProjAtomType, proj_anglr_m = self.ProjAnglrM)
        self.active_func = paras.active_func
        self.env_neurons = [1] + paras.Envnet
        self.env_out = paras.Envout
        # rij + env(Li + Lj)
        self.bond_neurons = [self.env_out * self.env_neurons[-1]+1] + paras.Bondnet
        # env(Li + Li)
        self.onsite_neurons = [self.env_out * self.env_neurons[-1]] + paras.onsite_net
        
        
        # load paras for train process.
        if self.istrain:
            self.num_epoch = paras.num_epoch
            self.batch_size = paras.batch_size
            self.valid_size = paras.valid_size
            self.start_learning_rate = paras.start_learning_rate
            self.decay_rate = paras.decay_rate
            self.decay_step = paras.decay_step
            self.trainmode = paras.trainmode
            self.savemodel = paras.savemodel
            self.display_epoch = paras.display_epoch
            self.sort_strength = paras.sort_strength
            self.sort_decay_alpha = np.log(self.sort_strength[1]/self.sort_strength[0])/self.num_epoch
            self.corr_strength = paras.corr_strength
            self.corr_decay_alpha = np.log(self.corr_strength[1]/self.corr_strength[0])/self.num_epoch
            
            
            self.withref = paras.withref
            if self.withref:
                self.refdir  = paras.refdir
                self.ref_ratio = paras.ref_ratio
            
            if self.trainmode.lower() == 'from_scratch':
                self.restart = False
            elif self.trainmode.lower() == 'restart':
                self.restart = True
                self.read_checkpoint = paras.read_checkpoint
            else:
                self.restart = False

            if self.savemodel:
                self.save_epoch = paras.save_epoch
                self.save_checkpoint = paras.save_checkpoint

        if self.istest or self.ispredict:
            self.batch_size = paras.batch_size
            self.restart = True
        self.read_checkpoint = paras.read_checkpoint
    

    def train(self):
        
        self.envnets = self.buildNN.BuildEnvNN(env_neurons=self.env_neurons,
                            afunc= self.active_func)
        self.bondnets = self.buildNN.BuildBondNN(bond_neurons= self.bond_neurons,
                            afunc=self.active_func)
        self.onsitenets = self.buildNN.BuildOnsiteNN(onsite_neurons=self.onsite_neurons,
                            afunc= self.active_func)

        self.bond_index_map  = self.buildNN.bond_index_map
        self.bond_num_hops = self.buildNN.bond_num_hops
        self.onsite_index_map = self.buildNN.onsite_index_map
        self.onsite_num = self.buildNN.onsite_num

        if self.restart:
            read_cp = th.load(self.read_checkpoint)
            for ikey in self.envnets.keys():
                self.envnets[ikey].load_state_dict(read_cp['env-'+ikey])
            for ikey in self.bondnets.keys():
                self.bondnets[ikey].load_state_dict(read_cp['bond-'+ikey])
            for ikey in self.onsitenets.keys():
                self.onsitenets[ikey].load_state_dict(read_cp['onsite-'+ikey])
            
        train_paras = []
        for ikey in self.envnets.keys():
            train_paras.append({'params':self.envnets[ikey].parameters()})
        
        for ikey in self.bondnets.keys():
            train_paras.append({'params':self.bondnets[ikey].parameters()})
        
        for ikey in self.onsitenets.keys():
            train_paras.append({'params':self.onsitenets[ikey].parameters()})

        optimizer = th.optim.Adam(train_paras, lr=self.start_learning_rate)  
        scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=self.decay_step, gamma = self.decay_rate)
        criterion = th.nn.MSELoss(reduction='mean')

        self.dataload.trainparas()
        if self.withref:
            self.dataload.perfdata()
            perfeigs = self.dataload.perfeigs
            pfeig_tensor = th.from_numpy(perfeigs).float()

        # epoch_used = 0
        for  epoch_used in range(self.num_epoch):
            set_id, tr_aseStructs, treigs, valid_id, valid_aseStructs, valid_eigs = \
                    self.dataload.traindata()

            treigs_tensor = th.from_numpy(treigs).float()
            trklist = self.dataload.setkpoints[set_id]
            nkps = len(trklist)
            nbandw_batch = []
            current_corr_strength = self.corr_strength[0] * np.exp(self.corr_decay_alpha * epoch_used)

            for istr in range(self.batch_size):
                self.structinput(tr_aseStructs[istr])
                norbs = np.sum(self.AtomNOrbs)
                nbanddft = treigs.shape[2]
                up_nband = min(nbanddft,norbs)

                ValElec = np.asarray(self.bondbuild.ProjValElec)
                NumElecs = np.sum(ValElec[self.bondbuild.TypeID])
                # when the NumElecs is odd number, the NumValband doesn's count the halp occupied band.
                NumValband = int(NumElecs // self.SpinDeg)
                NumKValbands = int(nkps * NumElecs // self.SpinDeg)

                assert NumValband <= up_nband

                if istr == 0:
                    #eig_batch = th.zeros([self.batch_size,nkps,norbs])
                    eig_batch = th.zeros([self.batch_size,nkps,up_nband])

                self.nnhoppings()
                self.SKhoppings()
                
                if self.correction_mode==2:
                    self.SKcorrection(corr_strength=current_corr_strength)
                elif self.correction_mode==1:
                    self.SKcorrection()

                self.HSmat(hoppings = self.hoppings_corr, overlaps = self.overlaps_corr , 
                                    onsiteEs = self.onsiteEs_corr, onsiteSs = self.onsiteSs_corr)
                eigks = self.Eigenvalues(kpoints = trklist)

                # Cal Fermi level (coarse, no need to be too much accurate) and band w.r.t. Fermi level:
                # just by  counting the num of electronic states = num of electrons.
                treigs_sort, sort_index = th.sort(th.reshape(treigs_tensor[istr],[-1]))
                trCoarseEfermi = (treigs_sort[NumKValbands] + treigs_sort[NumKValbands-1])/2.0
                treigs_tensor[istr] -=  trCoarseEfermi
                label_eigs = treigs_tensor[istr,:,0:up_nband]
                
                esort,sort_index = th.sort(th.reshape(eigks,[-1]))
                CoarseEfermi = (esort[NumKValbands]  + esort[NumKValbands-1])/2.0
                eigks -= CoarseEfermi
                eigks = eigks[:,0:up_nband]

                if self.use_E_win: 
                    out_index = label_eigs > self.energy_max
                    in_index = label_eigs < self.energy_max
                    in_num_kband = int(th.sum(in_index).numpy())
                    nbandwistr = int(np.ceil(in_num_kband/nkps))
                    nbandw_batch.append(nbandwistr)
                    # to remove the difference b.w. predicted eigs and labels. since the out window eigs is not considered for training.  d
                    eigks[out_index] = label_eigs[out_index]    
                
                eig_batch[istr] = eigks
            
            # note, the training band index star at 0
            if self.use_E_win: 
                # determine the band window for different traing data sets with different system size.
                band_max = np.min(nbandw_batch)
            elif self.use_I_win:
                band_max = self.band_max
            else:
                 print('error in define a window for trainng eigenvalues ')
            assert up_nband >= band_max
            band_min = self.band_min
            nbandw = band_max - band_min
            
            trewindow = th.reshape(treigs_tensor[:,:,band_min:band_max],[-1,nbandw])
            nnewindow = th.reshape(eig_batch[:,:,band_min:band_max],[-1,nbandw])
            
            current_strength = self.sort_strength[0] * np.exp(self.sort_decay_alpha * epoch_used)            
            trssort = torchsort.soft_sort(values = trewindow , regularization_strength=current_strength, regularization="l2")
            nnssort = torchsort.soft_sort(values = nnewindow , regularization_strength=current_strength, regularization="l2")
    
            trssort = th.reshape(trssort, [self.batch_size, nkps, nbandw])
            nnssort = th.reshape(nnssort, [self.batch_size, nkps, nbandw])

            losstr = criterion(trssort, nnssort)

            if nkps > 1:
                # randon choose nk_diff kps' eigenvalues to gen Delta eig.
                # nk_diff = max(nkps//4,1)     
                nk_diff = nkps        
                k_diff_i = np.random.choice(nkps,nk_diff,replace=False)
                k_diff_j = np.random.choice(nkps,nk_diff,replace=False)
                while (k_diff_i==k_diff_j).all():
                    k_diff_j = np.random.choice(nkps,nk_diff,replace=False)
                tr_eig_diff = trssort[:,k_diff_i,:] - trssort[:,k_diff_j,:]
                nn_eig_diff = nnssort[:,k_diff_i,:] - nnssort[:,k_diff_j,:]

                loss_diff_e =  criterion(tr_eig_diff,nn_eig_diff) 
                
                losstr = (losstr + loss_diff_e)/2.0
        

            if self.withref:
                self.structinput(self.dataload.perfstruct)
                nkps = len(self.dataload.perfkpionts)
                norbs = np.sum(self.AtomNOrbs)
                if len(pfeig_tensor.shape)==3:
                    # [nsnap,nk,nband]
                    nbanddft = pfeig_tensor.shape[2]
                elif len(pfeig_tensor.shape)==2:
                    # [nk,nband], nsnap is 1
                    nbanddft = pfeig_tensor.shape[1]
                elif len(pfeig_tensor.shape)==1:
                    # [nband], nsnap and nk both 1.
                    nbanddft = pfeig_tensor.shape[0]
                else:
                    print('Error in ref eigs shape.')

                up_nband = min(nbanddft,norbs)
                ValElec = np.asarray(self.bondbuild.ProjValElec)
                NumElecs = np.sum(ValElec[self.bondbuild.TypeID])
                # when the NumElecs is odd number, the NumValband doesn's count the halp occupied band.
                NumValband = int(NumElecs // self.SpinDeg)
                NumKValbands = int(nkps * NumElecs // self.SpinDeg)
                assert NumValband <= up_nband

                self.nnhoppings()
                self.SKhoppings()
                self.SKcorrection()
                self.HSmat(hoppings = self.hoppings_corr, overlaps = self.overlaps_corr , 
                                    onsiteEs = self.onsiteEs_corr, onsiteSs = self.onsiteSs_corr)
                eig_perf = self.Eigenvalues(kpoints = self.dataload.perfkpionts)
                
                rfeigs_sort, sort_index = th.sort(th.reshape(pfeig_tensor,[-1]))
                rfCoarseEfermi = (rfeigs_sort[NumKValbands] + rfeigs_sort[NumKValbands-1])/2.0
                pfeig_tensor -=  rfCoarseEfermi

                label_eigs = pfeig_tensor[:,0:up_nband]
                out_index = label_eigs > self.energy_max
                in_index = label_eigs < self.energy_max
                in_num_kband = int(th.sum(in_index).numpy())
                band_max = int(np.ceil(in_num_kband/nkps))

                # Cal Fermi level (coarse, no need to be too much accurate) and band w.r.t. Fermi level:
                esort,sort_index = th.sort(th.reshape(eig_perf,[-1]))
                CoarseEfermi = (esort[NumKValbands]  + esort[NumKValbands-1])/2.0
                eig_perf -= CoarseEfermi
                eig_perf = eig_perf[:,0:up_nband]
                eig_perf[out_index] = label_eigs[out_index]   

                band_min = self.band_min
                nbandw = band_max - band_min

                rfewindow = th.reshape(pfeig_tensor[:,band_min:band_max],[-1,nbandw])
                nnrfewindow = th.reshape(eig_perf[:,band_min:band_max],[-1,nbandw])


                rfssort = torchsort.soft_sort(values = rfewindow , regularization_strength=current_strength, regularization="l2")
                nnrfssort = torchsort.soft_sort(values = nnrfewindow , regularization_strength=current_strength, regularization="l2")
                
                rfssort = th.reshape(rfssort, [nkps, nbandw])
                nnrfssort = th.reshape(nnrfssort, [nkps, nbandw])
                
                losspf = criterion(rfssort,nnrfssort)
                if nkps > 1:
                    # randon choose nk_diff kps' eigenvalues to gen Delta eig.
                    # nk_diff = max(nkps//4,1)     
                    nk_diff = nkps        
                    k_diff_i = np.random.choice(nkps,nk_diff,replace=False)
                    k_diff_j = np.random.choice(nkps,nk_diff,replace=False)
                    while (k_diff_i==k_diff_j).all():
                        k_diff_j = np.random.choice(nkps,nk_diff,replace=False)
                    rf_eig_diff = rfssort[k_diff_i,:] - rfssort[k_diff_j,:]
                    nnrf_eig_diff = nnrfssort[k_diff_i,:] - nnrfssort[k_diff_j,:]
                    loss_pf_diff_e =  criterion(rf_eig_diff,nnrf_eig_diff) 
                    
                    losspf = (losspf + loss_pf_diff_e)/2.0

                loss = self.ref_ratio * losspf + (1-self.ref_ratio) * losstr
            
            else:
                loss = losstr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            currentlr = optimizer.state_dict()['param_groups'][0]['lr']
            scheduler.step()

            if epoch_used % self.display_epoch == 0:
                # calculate validation error.
                valid_klist = self.dataload.setkpoints[valid_id]
                vldeigs_tensor = th.from_numpy(valid_eigs).float()
                nkps = len(valid_klist)
                nbandw_batch = []
                for istr in range(self.valid_size):
                    self.structinput(valid_aseStructs[istr])
                    norbs = np.sum(self.AtomNOrbs)
                    nbanddft = valid_eigs.shape[2]
                    up_nband = min(nbanddft,norbs)
                    
                    ValElec = np.asarray(self.bondbuild.ProjValElec)
                    NumElecs = np.sum(ValElec[self.bondbuild.TypeID])
                    NumValband = int(NumElecs // self.SpinDeg)
                    # when the NumElecs is odd number, the NumValband doesn's count the halp occupied band.
                    NumValband = int(NumElecs // self.SpinDeg)
                    NumKValbands = int(nkps * NumElecs // self.SpinDeg)

                    if istr == 0:
                        # eig_batch = th.zeros([self.batch_size, nkps, norbs])
                        eig_batch = th.zeros([self.batch_size, nkps, up_nband])

                    self.nnhoppings()
                    self.SKhoppings()
                    self.SKcorrection()
                    self.HSmat(hoppings = self.hoppings_corr, overlaps = self.overlaps_corr , 
                                    onsiteEs = self.onsiteEs_corr, onsiteSs = self.onsiteSs_corr)
                    eigks = self.Eigenvalues(kpoints = valid_klist)
                      
                    # Cal Fermi level (coarse, no need to be too much accurate) and band w.r.t. Fermi level:
                    esort,sort_index = th.sort(th.reshape(eigks,[-1]))
                    CoarseEfermi = (esort[nkps * NumValband]  + esort[nkps * NumValband-1])/2.0
                    eigks -= CoarseEfermi

                    eigks = eigks[:,0:up_nband]
                    
                    vldeigs_tensor_sort,sort_index = th.sort(th.reshape(vldeigs_tensor[istr],[-1]))
                    vlCoarseEfermi = (vldeigs_tensor_sort[nkps * NumValband] + vldeigs_tensor_sort[nkps * NumValband-1])/2.0
                    vldeigs_tensor[istr] -=  vlCoarseEfermi

                    label_eigs = vldeigs_tensor[istr,:,0:up_nband]
                    if self.use_E_win: 
                        out_index = label_eigs > self.energy_max
                        in_index  = label_eigs < self.energy_max
                        in_num_kband = int(th.sum(in_index).numpy())
                        nbandwistr = int(np.ceil(in_num_kband/nkps))
                        nbandw_batch.append(nbandwistr)
                        eigks[out_index] = label_eigs[out_index]   

                    eig_batch[istr] = eigks
                if self.use_E_win: 
                    band_max = np.min(nbandw_batch)
                elif self.use_I_win:
                    band_max = self.band_max
                else:
                    print('error in define a window for trainng eigenvalues ')
                assert up_nband >= band_max
                # for validation, do not use soft sort.
                band_min = self.band_min
                nbandw = band_max - band_min
                valid_err = criterion(vldeigs_tensor[:,:,band_min:band_max], eig_batch[:,:,band_min:band_max])

                if epoch_used == 0:
                    print('--'*48)
                    print('#  epoch_used       tr-err           valid-err          lr          sort_r')
                    print('--'*48)
                #print('#\t', epoch_used, '\t\t{:0.4g}'.format(loss.item()),' \t\t{:0.4g}'.format(currentlr))
                print(f'{epoch_used:6d} {np.sqrt(loss.item()):16.6g} {np.sqrt(valid_err.item()):16.6g}  {currentlr:16.4g} {current_strength:16.4g}')

            if self.savemodel and epoch_used % self.save_epoch == 0:
                modeldict = {}
                for ikey in self.envnets.keys():
                    modeldict['env-'+ikey] = self.envnets[ikey].state_dict()
                for ikey in self.bondnets.keys():
                    modeldict['bond-'+ikey] = self.bondnets[ikey].state_dict()
                for ikey in self.onsitenets.keys():
                    modeldict['onsite-'+ikey] = self.onsitenets[ikey].state_dict()
                th.save(modeldict,self.save_checkpoint)


    def loadmodel(self):
        self.envnets = self.buildNN.BuildEnvNN(env_neurons=self.env_neurons,
                            afunc= self.active_func)
        self.bondnets = self.buildNN.BuildBondNN(bond_neurons= self.bond_neurons,
                            afunc=self.active_func)
        self.onsitenets = self.buildNN.BuildOnsiteNN(onsite_neurons=self.onsite_neurons,
                            afunc= self.active_func)

        self.bond_index_map  = self.buildNN.bond_index_map
        self.bond_num_hops = self.buildNN.bond_num_hops
        self.onsite_index_map = self.buildNN.onsite_index_map
        self.onsite_num = self.buildNN.onsite_num

        read_cp = th.load(self.read_checkpoint)
        for ikey in self.envnets.keys():
            self.envnets[ikey].load_state_dict(read_cp['env-'+ikey])
        for ikey in self.bondnets.keys():
            self.bondnets[ikey].load_state_dict(read_cp['bond-'+ikey])
        for ikey in self.onsitenets.keys():
            self.onsitenets[ikey].load_state_dict(read_cp['onsite-'+ikey])


    def test(self):
        self.envnets = self.buildNN.BuildEnvNN(env_neurons=self.env_neurons,
                            afunc= self.active_func)
        self.bondnets = self.buildNN.BuildBondNN(bond_neurons= self.bond_neurons,
                            afunc=self.active_func)
        self.onsitenets = self.buildNN.BuildOnsiteNN(onsite_neurons=self.onsite_neurons,
                            afunc= self.active_func)

        self.bond_index_map  = self.buildNN.bond_index_map
        self.bond_num_hops = self.buildNN.bond_num_hops
        self.onsite_index_map = self.buildNN.onsite_index_map
        self.onsite_num = self.buildNN.onsite_num
        
        read_cp = th.load(self.read_checkpoint)
        for ikey in self.envnets.keys():
            self.envnets[ikey].load_state_dict(read_cp['env-'+ikey])
        for ikey in self.bondnets.keys():
            self.bondnets[ikey].load_state_dict(read_cp['bond-'+ikey])
        for ikey in self.onsitenets.keys():
            self.onsitenets[ikey].load_state_dict(read_cp['onsite-'+ikey])
        
        criterion = th.nn.MSELoss(reduction='mean')

        ts_aseStructs, tseigs, tskpoints = self.dataload.testdata()
        tseigs_tensor = th.from_numpy(tseigs).float()
        assert len(tskpoints.shape) == 2
        nkps = len(tskpoints)
        numsnaps = len(ts_aseStructs)
        nbandw_batch = []
        for istr in range(numsnaps):
            self.structinput(ts_aseStructs[istr])
            norbs = np.sum(self.AtomNOrbs)
            nbanddft = tseigs.shape[2]
            up_nband = min(nbanddft,norbs)
            ValElec = np.asarray(self.bondbuild.ProjValElec)
            NumElecs = np.sum(ValElec[self.bondbuild.TypeID])
            NumValband = int(NumElecs // self.SpinDeg)
            NumKValbands = int(nkps * NumElecs // self.SpinDeg)
            assert NumValband <= up_nband
            if istr == 0:
                #eig_batch = th.zeros([self.batch_size,nkps,norbs])
                eig_batch = th.zeros([numsnaps,nkps,up_nband])
                tsfermi = th.zeros([numsnaps,1])
                #if self.saveeigs:
                ts_nn_eigs = th.zeros([numsnaps,nkps,norbs])

            self.nnhoppings()
            self.SKhoppings()
            self.SKcorrection()
            self.HSmat(hoppings = self.hoppings_corr, overlaps = self.overlaps_corr , 
                                    onsiteEs = self.onsiteEs_corr, onsiteSs = self.onsiteSs_corr)
            eigks = self.Eigenvalues(kpoints = tskpoints)

            tseigs_sort, sort_index = th.sort(th.reshape(tseigs_tensor[istr],[-1]))
            tsCoarseEfermi = (tseigs_sort[NumKValbands] + tseigs_sort[NumKValbands-1])/2.0
            tseigs_tensor[istr] -=  tsCoarseEfermi
            tsfermi[istr] = tsCoarseEfermi

            esort,sort_index = th.sort(th.reshape(eigks,[-1]))
            CoarseEfermi = (esort[NumKValbands]  + esort[NumKValbands-1])/2.0
            eigks -= CoarseEfermi

            #if self.saveeigs:
            ts_nn_eigs[istr] = eigks

            label_eigs = tseigs_tensor[istr,:,0:up_nband]
            eigks_upband = eigks[:,0:up_nband]

            if self.use_E_win:
                out_index = label_eigs > self.energy_max
                in_index = label_eigs < self.energy_max
                in_num_kband = int(th.sum(in_index).numpy())
                nbandwistr = int(np.ceil(in_num_kband/nkps))
                nbandw_batch.append(nbandwistr)
                eigks_upband[out_index] = label_eigs[out_index] 

            eig_batch[istr] = eigks_upband

            if (istr+1)%self.batch_size==0:
                ibatch = (istr+1)//self.batch_size
                assert ibatch > 0
                ist = (ibatch - 1) * self.batch_size
                ied = (ibatch) * self.batch_size
                
                if self.use_E_win: 
                    band_max = np.min(np.asarray(nbandw_batch)[ist:ied])
                elif self.use_I_win:
                    band_max = self.band_max
                
                assert up_nband >= band_max
                band_min = self.band_min
                nbandw = band_max - band_min

                tsewindow = th.reshape(tseigs_tensor[ist:ied,:,band_min:band_max],[-1,nbandw])
                nnewindow = th.reshape(eig_batch[ist:ied,:,band_min:band_max],[-1,nbandw])
                ts_batch_err = criterion(tsewindow, nnewindow)
                if ibatch ==0:
                    print('--'*48)
                    print('#  batch_test       ts-err')
                    print('--'*48)
                print(f'{ibatch:6d} {np.sqrt(ts_batch_err.item()):16.6g}')

        if self.use_E_win: 
            band_max = np.min(nbandw_batch)
        elif self.use_I_win:
            band_max = self.band_max - 0
        assert up_nband >= band_max
        band_min = self.band_min
        nbandw = band_max - band_min

        tsewindow = th.reshape(tseigs_tensor[:,:,band_min:band_max],[-1,nbandw])
        nnewindow = th.reshape(eig_batch[:,:,band_min:band_max],[-1,nbandw])
        
        ts_err = criterion(tsewindow, nnewindow)
        print(f'total error : {np.sqrt(ts_err.item()):16.6g}')
        np.save(self.dataload.testdir + '/nneigs.npy', ts_nn_eigs.detach().numpy())
        np.save(self.dataload.testdir + '/tsfermi.npy', tsfermi.detach().numpy())


    def structinput(self,struct,TRsymm=True):
        # initialize the env and bond class for given struct.
        self.bondbuild.BondStuct(struct)
        self.bondbuild.GetBonds(TRsymm=TRsymm)
        self.envbuild.IniUsingAse(struct)
        self.envbuild.Projection()

        self.Bonds = self.bondbuild.Bonds
        self.BondsOnSite = self.bondbuild.BondsOnSite
        self.bondtype = self.bondbuild.Uniqsybl
        self.envtype = self.envbuild.Uniqsybl
        self.bondtypeid = self.bondbuild.TypeID
        self.envtypeid = self.envbuild.TypeID
        self.AtomNOrbs = self.bondbuild.AtomNOrbs 
        self.AtomTypeNOrbs = self.bondbuild.AtomTypeNOrbs

    def nnhoppings(self):
        """ Generate the nn corrections of hoppings and onsite Es.
        
        Attributes:
        self.nn_hoppings, self.nn_onsiteEs, the nn correction of hoppings and on
        """
        self.nn_hoppings = []
        for ib in range(len(self.Bonds)):
            ibond = self.Bonds[ib]
            itypeid = self.bondtypeid[ibond[0]]
            ibtype = self.bondtype[itypeid]
            jtypeid = self.bondtypeid[ibond[1]]
            jbtype = self.bondtype[jtypeid]
            if itypeid > jtypeid:
                bondname = jbtype + ibtype
            else:
                bondname = ibtype + jbtype
               
            rij = (self.bondbuild.Positions[ibond[1]] 
                    - self.bondbuild.Positions[ibond[0]] 
                        + np.dot(ibond[2:], self.bondbuild.Lattice))
            
            rrij = np.linalg.norm(rij)
            emdenvib = self.EnvCoding(ibond)

            rij_tensor = th.from_numpy(np.asarray([[1/(rrij)]])).float()
            emdenvib = th.cat([rij_tensor, emdenvib],axis=1)

            self.nn_hoppings.append(th.reshape(self.bondnets[bondname](emdenvib),[-1]))
            #if ib==0:
            #    bond_env = emdenvib
            #else:
            #    bond_env = th.cat([bond_env,emdenvib])
        self.nn_onsiteEs=[]
        for isite in range(len(self.BondsOnSite)):
            ibond = self.BondsOnSite[isite]
            sitetypeid = self.bondtypeid[ibond[0]]
            sitename = self.bondtype[sitetypeid]

            emdenvib = self.EnvCoding(ibond)
            if isite == 0:
                onsite_env = emdenvib
            else:
                onsite_env = th.cat([onsite_env,emdenvib])

            self.nn_onsiteEs.append(th.reshape(self.onsitenets[sitename](emdenvib),[-1]))


    def SKhoppings(self):
        """ using SK files to get the emperical TB paras. 

        Attributes:
            self.onsiteEs, self.onsiteSs : the onsite energies and overlaps.
            self.skhoppings and self.skoverlaps: the hoppings and overlaps between sites.
        """
        # self.bond_index_map  = self.buildNN.bond_index_map
        # self.bond_num_hops = self.buildNN.bond_num_hops
        # self.onsite_index_map = self.buildNN.onsite_index_map
        # self.onsite_num = self.buildNN.onsite_num

        if not hasattr(self,'bond_index_map'):
            self.bond_index_map, self.bond_num_hops = self.IndMap.Bond_Ind_Mapings()

        if not hasattr(self,'onsite_index_map'): 
             self.onsite_index_map, self.onsite_num = self.IndMap.Onsite_Ind_Mapings()
        # print('HHHHHH')
        # print(self.bond_index_map)

        self.onsiteEs = []
        self.onsiteSs = []
 
        for ib in range(len(self.BondsOnSite)):
            ibond  = self.BondsOnSite[ib]
            iatype, jatype = self.bondtypeid[ibond[0]] , self.bondtypeid[ibond[1]]
            assert iatype == jatype, "i type should equal j type."
            iatypesbl = self.bondtype[iatype]
            num_onsite = self.onsite_num[iatypesbl]       

            siteE = np.zeros([num_onsite])
            siteS = np.zeros([num_onsite])      
            for ish in self.ProjAnglrM[iatype]:     # ['s','p',..]
                shidi = self.AnglrMID[ish]          # 0,1,2,...   
                indx = self.onsite_index_map[iatypesbl][ish]
                siteE[indx] = self.SKIntIns.SiteE[iatype][shidi]
                siteS[indx] = 1.0
               
            # self.onsiteEs.append(siteE)
            # self.onsiteSs.append(siteS)

            self.onsiteEs.append(th.from_numpy(siteE).float())
            self.onsiteSs.append(th.from_numpy(siteS).float())


        # get hopping parameters.    
        self.skhoppings = []
        self.skoverlaps = []
        for ib in range(len(self.Bonds)):
            
            ibond = self.Bonds[ib]
            dirvec = (self.bondbuild.Positions[ibond[1]] 
                       - self.bondbuild.Positions[ibond[0]] 
                        + np.dot(ibond[2:], self.bondbuild.Lattice))
            dist = np.linalg.norm(dirvec)
            dist = dist/self.au2Ang      # the sk files is written in atomic unit
            iatype, jatype = self.bondtypeid[ibond[0]] , self.bondtypeid[ibond[1]]
            iatypesbl, jatypesbl = self.bondtype[iatype], self.bondtype[jatype]

            HKinterp12 = self.SKIntIns.IntpSK(itype=iatype,jtype=jatype,dist=dist)
            """
            for a A-B bond, there are two sk files, A-B and B-A.
            e.g.:
                sp hopping, A-B: sp. means A(s) - B(p) hopping. 
                we know, A(s) - B(p) => B(p)-A(s)
                therefore, from A-B sp, we know B-A ps.
            """
            
            if iatype == jatype:
                # HKinterp12 = SKIntIns.IntpSK(itype=iatype,jtype=jatype,dist=dist)
                # view, the same addr. in mem.
                HKinterp21 = HKinterp12
            else:
                # HKinterp12 = SKIntIns.IntpSK(itype=iatype,jtype=jatype,dist=dist)
                HKinterp21 = self.SKIntIns.IntpSK(itype=jatype,jtype=iatype,dist=dist)
                
            # hoppings = {}
            # overlaps = {}
            num_hops = self.bond_num_hops[iatypesbl + jatypesbl]

            #if iatype <= jatype:
            #    bondname = iatypesbl+jatypesbl
            #else:
            #    bondname = jatypesbl+iatypesbl
            bondname = iatypesbl+jatypesbl

            hoppings = np.zeros([num_hops])
            overlaps = np.zeros([num_hops])

            for ish in self.ProjAnglrM[iatype]:
                shidi = self.AnglrMID[ish]
                # norbi = 2*shidi+1

                for jsh in self.ProjAnglrM[jatype]:
                    shidj = self.AnglrMID[jsh]
                    # norbj = 2 * shidj + 1

                    if shidi < shidj:
                        Hvaltmp = HKinterp12[self.SKAnglrMHSID[ish+jsh]]
                        Svaltmp = HKinterp12[self.SKAnglrMHSID[ish+jsh] + self.NumHvals]
                    else:
                        Hvaltmp = HKinterp21[self.SKAnglrMHSID[ish+jsh]]
                        Svaltmp = HKinterp21[self.SKAnglrMHSID[ish+jsh] + self.NumHvals]
                    # print('aAAAAAA')
                    # print(iatype, jatype, self.ProjAnglrM,ish,jsh)
                    # print(bondname, self.bond_index_map[bondname])
                    indx = self.bond_index_map[bondname][ish+jsh]
                    hoppings[indx] = Hvaltmp
                    overlaps[indx] = Svaltmp
            
            # self.skhoppings.append(hoppings)
            # self.skoverlaps.append(overlaps)    
            self.skhoppings.append(th.from_numpy(hoppings).float())
            self.skoverlaps.append(th.from_numpy(overlaps).float())   

    def SKcorrection(self,corr_strength=1):
        """Add the nn correction to SK parameters hoppings and onsite Es.
        
        Note: the overlaps are fixed on changed of SK parameters.
        """
        self.onsiteEs_corr = []
        self.onsiteSs_corr = []
        for ib in range(len(self.BondsOnSite)):
            # onsiteEs_th = th.from_numpy(self.onsiteEs[ib]).float()
            onsiteEs_th = self.onsiteEs[ib]
            onsiteEs_th.requires_grad = False
            if self.correction_mode == 1:
                self.onsiteEs_corr.append(onsiteEs_th * (1 + self.nn_onsiteEs[ib]))
            # self.onsiteEs_corr.append(self.nn_onsiteEs[ib] + onsiteEs_th)
            elif self.correction_mode == 2:
                self.onsiteEs_corr.append(onsiteEs_th + corr_strength * self.nn_onsiteEs[ib])
            # onsiteSs_th = th.from_numpy(self.onsiteSs[ib]).float()
            onsiteSs_th = self.onsiteSs[ib]
            onsiteSs_th.requires_grad = False
            # no correction to overlap S, just transform to tensor.
            self.onsiteSs_corr.append(onsiteSs_th)
    
        self.hoppings_corr = []
        self.overlaps_corr = []
        for ib in range(len(self.Bonds)):
            if np.linalg.norm(self.skhoppings[ib]) < 1e-6:
                # hoppings_th = th.from_numpy(self.skhoppings[ib] + 1e-6).float()
                hoppings_th = self.skhoppings[ib] + 1e-6
            else:
                # hoppings_th = th.from_numpy(self.skhoppings[ib] + 1e-6).float()
                hoppings_th = self.skhoppings[ib] + 1e-6
            hoppings_th.requires_grad= False

            if self.correction_mode == 1:
                self.hoppings_corr.append(hoppings_th * (1 + self.nn_hoppings[ib]))
            elif self.correction_mode == 2:
                self.hoppings_corr.append(hoppings_th + corr_strength * self.nn_hoppings[ib])
            
            # overlaps_th = th.from_numpy(self.skoverlaps[ib]).float()
            overlaps_th = self.skoverlaps[ib]
            overlaps_th.requires_grad = False
            # no correction to overlaps, just transform to tensor.
            self.overlaps_corr.append(overlaps_th)

    
    def HSmat(self, hoppings, overlaps, onsiteEs, onsiteSs):
        """using the sk format hoppings, overlaps and onsiteEs Ss, to build H-Hamiltonian and S-overlap matrix.
        
        Args:
            hoppings,overlaps, onsiteEs, onsiteSs:
                the TB parameters to get in SK format, (sk or nn+sk).
        """
        self.BondSBlock = []
        self.BondHBlock = []
        for ib in range(len(self.BondsOnSite)):
            ibond  = self.BondsOnSite[ib]
            iatype, jatype = self.bondtypeid[ibond[0]] , self.bondtypeid[ibond[1]]
            assert iatype == jatype, "i type should equal j type."

            iatypesbl = self.bondtype[iatype]

            Hamilblock = th.zeros([self.AtomTypeNOrbs[iatype], self.AtomTypeNOrbs[jatype]])
            Soverblock = th.zeros([self.AtomTypeNOrbs[iatype], self.AtomTypeNOrbs[jatype]])

            ist = 0
            for ish in self.ProjAnglrM[iatype]:     # ['s','p',..]
                shidi = self.AnglrMID[ish]          # 0,1,2,...  
                norbi = 2*shidi + 1 
                indx = self.onsite_index_map[iatypesbl][ish]
                Hamilblock[ist:ist+norbi, ist:ist+norbi] = th.eye(norbi) * onsiteEs[ib][indx]
                Soverblock[ist:ist+norbi, ist:ist+norbi] = th.eye(norbi) * onsiteSs[ib][indx]
                ist = ist +norbi

            self.BondHBlock.append(Hamilblock)
            self.BondSBlock.append(Soverblock)

        for ib in range(len(self.Bonds)):
            ibond = self.Bonds[ib]
            dirvec = (self.bondbuild.Positions[ibond[1]] 
                       - self.bondbuild.Positions[ibond[0]] 
                        + np.dot(ibond[2:], self.bondbuild.Lattice))
            dist = np.linalg.norm(dirvec)
            dirvec = dirvec/dist

            iatype, jatype = self.bondtypeid[ibond[0]] , self.bondtypeid[ibond[1]]
            iatypesbl, jatypesbl = self.bondtype[iatype], self.bondtype[jatype]

            Hamilblock = th.zeros([self.AtomTypeNOrbs[iatype], self.AtomTypeNOrbs[jatype]])
            Soverblock = th.zeros([self.AtomTypeNOrbs[iatype], self.AtomTypeNOrbs[jatype]])

            
            #if iatype <= jatype:
            #    bondname = iatypesbl+jatypesbl
            #else:
            #    bondname = jatypesbl+iatypesbl
            bondatomtype = iatypesbl+jatypesbl
            
            ist = 0
            for ish in self.ProjAnglrM[iatype]:
                shidi = self.AnglrMID[ish]
                norbi = 2*shidi+1
                
                jst = 0
                for jsh in self.ProjAnglrM[jatype]:
                    shidj = self.AnglrMID[jsh]
                    norbj = 2 * shidj + 1   

                    idx= self.bond_index_map[bondatomtype][ish+jsh]
                    if shidi < shidj:
                        tmpH = self.rotHS.RotHS(Htype=ish+jsh, Hvalue=hoppings[ib][idx], Angvec=dirvec)
                        # Hamilblock[ist:ist+norbi, jst:jst+norbj] = th.transpose(tmpH,dim0=0,dim1=1)
                        Hamilblock[ist:ist+norbi, jst:jst+norbj] = (-1.0)**(shidi + shidj) * th.transpose(tmpH,dim0=0,dim1=1)
                        tmpS = self.rotHS.RotHS(Htype=ish+jsh, Hvalue=overlaps[ib][idx], Angvec=dirvec)
                        # Soverblock[ist:ist+norbi, jst:jst+norbj] = th.transpose(tmpS,dim0=0,dim1=1)
                        Soverblock[ist:ist+norbi, jst:jst+norbj] = (-1.0)**(shidi + shidj) * th.transpose(tmpS,dim0=0,dim1=1)

                    else:
                        tmpH = self.rotHS.RotHS(Htype=jsh+ish, Hvalue=hoppings[ib][idx], Angvec=dirvec)
                        Hamilblock[ist:ist+norbi, jst:jst+norbj] = tmpH

                        tmpS = self.rotHS.RotHS(Htype=jsh+ish, Hvalue = overlaps[ib][idx], Angvec = dirvec)
                        Soverblock[ist:ist+norbi, jst:jst+norbj] = tmpS
                
                    jst = jst + norbj 
                ist = ist + norbi   
                        #print( iatype, jatype, Soverblock.shape)
            self.BondSBlock.append(Soverblock)
            self.BondHBlock.append(Hamilblock)
        self.allBonds = np.concatenate([self.BondsOnSite,self.Bonds],axis=0)

    def Hamilreal2K(self, hij_all, bond, kpath, num_orbs,TRsymm=True):
        """ transfer H(r) to H(k).

        Args:
            hij_all : all the Hblocks and Sblocks.
            bond : all the bond [i j Rx Ry Rz]. 
            kpath : kpoints lists.
            num_orbs ：a list of the total orbitals on each atom. eg. atom with s and p orbitals : 4.

        return:
            H(k) or S(k) depending on the hij_all = Hblocks or Sblocks.
        """
        total_orbs = np.sum(num_orbs)
        Hk= th.zeros([len(kpath), total_orbs, total_orbs], dtype = th.complex64)
        for ik in range(len(kpath)):
            k = kpath[ik]
            hk = th.zeros([total_orbs,total_orbs],dtype = th.complex64)
            for ib in range(len(bond)):
                R = bond[ib,2:]
                i = bond[ib,0]
                j = bond[ib,1]

                ist = int(np.sum(num_orbs[0:i]))
                ied = int(np.sum(num_orbs[0:i+1]))
                jst = int(np.sum(num_orbs[0:j]))
                jed = int(np.sum(num_orbs[0:j+1]))
                if ib < len(num_orbs):
                    # len(num_orbs)= numatoms.
                    # the first numatoms are onsite energies.
                    if TRsymm:
                        hk[ist:ied,jst:jed] += 0.5 * hij_all[ib] * np.exp(-1j * 2 * np.pi* np.dot(k,R)) 
                    else:
                        hk[ist:ied,jst:jed] += hij_all[ib] * np.exp(-1j * 2 * np.pi* np.dot(k,R)) 
                else:
                    hk[ist:ied,jst:jed] += hij_all[ib] * np.exp(-1j * 2 * np.pi* np.dot(k,R)) 
            if TRsymm:
                hk = hk + hk.T.conj()

            #for i in range(total_orbs):
            #    hk[i,i] = hk[i,i] * 0.5
            Hk[ik] = hk
        return Hk


    def Eigenvalues(self,kpoints,TRsymm=True):
        """ using the tight-binding H and S matrix calculate eigenvalues at kpoints.
        
        Args:
            kpoints: the k-kpoints used to calculate the eigenvalues.
        Note: must have the BondHBlock and BondSBlock 
        """
        hkmat =  self.Hamilreal2K(hij_all = self.BondHBlock,bond = self.allBonds, 
                                    kpath=kpoints, num_orbs = self.AtomNOrbs,TRsymm=TRsymm)
        skmat =  self.Hamilreal2K(hij_all = self.BondSBlock,bond = self.allBonds, 
                                    kpath=kpoints, num_orbs = self.AtomNOrbs,TRsymm=TRsymm)
        
        # eig=[]
        # for i in range(len(kpoints)):
        #    chklowt = th.linalg.cholesky(skmat[i])
        #    chklowtinv = th.linalg.inv(chklowt)
            #The eigenvalues  in eig_k are returned in ascending order.
        #    eig_k = th.linalg.eigvalsh(chklowtinv @ hkmat[i] @chklowtinv.T.conj())
            #eig_k,eig_vec= th.linalg.eigvalsh(hkmat[i],skmat[i])
        #    eig.append(eig_k * 13.605662285137 * 2)
        
        chklowt = th.linalg.cholesky(skmat)
        chklowtinv = th.linalg.inv(chklowt)
        Heff = (chklowtinv @ hkmat @ th.transpose(chklowtinv,dim0=1,dim1=2).conj())
        eigks = th.linalg.eigvalsh(Heff) * 13.605662285137 * 2

        return eigks


    def EigCoding(self,eig,eig_hat):
        """ encoding eigenvalues."""
        # not used now.
        batch_size, nkp, nband = eig.shape
        Emin, Emax = self.energy_min, self.energy_max
        #sigma = (Emax - Emin)/nband
        sigma = (Emax - Emin)/64
        # x0 = th.linspace(Emin,Emax,nband)
        gs_inx = th.reshape(eig,[batch_size, nkp, 1, nband]) -  th.reshape(eig_hat,[batch_size, nkp,nband,1])
        gs_nn = th.exp((-1*gs_inx**2)/(2*sigma**2))
        filt_eig = th.sum(th.reshape(eig,[batch_size, nkp, 1, nband]) * gs_nn,axis=3)
        gscoff =  th.sum(gs_nn,axis=3)
        return filt_eig, gscoff
        

    def EnvCoding(self,ibond):
        """local environment coding to ensure symmetries, translation, rotation, permutation.
        
        Args:
            ibond: bond information obtained in BondBuild Class. 
                    [i,j,Rx,Ry,Rz]. ij refers the index of center of orbitals, 
                    R is the lattace vector in fractional coor. (Rx,Ry,Rz are integers.)\
        
        return:
            local environment corresponding to the ibond.
        """
        envib = self.envbuild.iBondEnv(ibond)
        envib_tensor = th.from_numpy(envib).float()
        ist=0
        for ii in range(2):
            typeid = self.bondtypeid[ibond[ii]]
            btype = self.bondtype[typeid]
            for it in range(len(self.envtype)):
                etype = self.envtype[it]
                ied = ist + self.envbuild.NumEnv[it]
                inputs = envib_tensor[ist:ied,0:1]
                xyz_scatter = self.envnets[btype+etype](inputs)
                if ii==0 and it==0:
                    xyz_scatter_1 = th.matmul(
                        th.transpose(envib_tensor[ist:ied,:],dim0=0,dim1=1),xyz_scatter)
                else:
                    xyz_scatter_1 += th.matmul(
                        th.transpose(envib_tensor[ist:ied,:],dim0=0,dim1=1),xyz_scatter)
                ist = ied

        xyz_scatter_1 = xyz_scatter_1/envib_tensor.shape[0]
        xyz_scatter_2 = xyz_scatter_1[:,0:self.env_out]
        emdenvib = th.matmul(
                        th.transpose(xyz_scatter_1,dim0=0,dim1=1),xyz_scatter_2)
        emdenvib = th.reshape(emdenvib,[1,-1])
        return emdenvib


