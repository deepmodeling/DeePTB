import warnings
import numpy as np
import ase 
import glob
from ase import Atoms
from ase.io import read, iread, write
from ase.io.trajectory import Trajectory

from sktb.StructData import StructBuild

class DataLoad(object):
    def __init__(self,paras) -> None:
        super().__init__()
        self.paras   = paras
        self.istrain = paras.istrain
        self.istest  = paras.istest
        self.ispredict = paras.ispredict  
        # file names
 
    def trainparas(self):
        self.xdatfile = self.paras.xdatfile
        self.eigfile  = self.paras.eigfile
        self.kpfile   = self.paras.kpfile
        self.trpath   = self.paras.train_data_path
        self.prefix   = self.paras.prefix
        self.valddir  = self.paras.valddir
        # self.format = paras.format

        self.num_epoch  = self.paras.num_epoch
        self.batch_size = self.paras.batch_size
        self.valid_size = self.paras.valid_size
        self.trdirs    = glob.glob(self.trpath + "/" + self.prefix + ".*")
        self.trdirs.append(self.valddir)

        self.numtrsets = len(self.trdirs)
        self.set_asetrajs = []
        self.set_dat_sizes = []
        self.setkpoints  = []
        self.seteigs     = []

        print('# Finding datadirs:')
        for ii in range(self.numtrsets):
            # aseAtoms = read(self.trdirs[ii] + '/' + self.xdatfile, 
            #                    format=self.format,index=':')
            # binary  trajectory  file in ase.io.trajectory object.
            asetrajs = Trajectory(filename=self.trdirs[ii] + '/' + self.xdatfile, mode='r')
            self.set_asetrajs.append(asetrajs)
            self.set_dat_sizes.append(len(asetrajs))
            kpionts = np.load(self.trdirs[ii] + '/' + self.kpfile)
            self.setkpoints.append(kpionts)
            eigs   = np.load(self.trdirs[ii] + '/' + self.eigfile)
            self.seteigs.append(eigs)

            # data info. out to log.
            print('#    - ' + self.trdirs[ii] + ': total snaps' + f'{len(asetrajs):6d}')

        self.index_count = 0
        self.set_index_count = np.zeros([self.numtrsets-1], dtype=int)
        
        set_ind=[]
        self.set_sample_szie = []
        for ii  in range(self.numtrsets-1):
            sample_ind = np.arange(0,self.set_dat_sizes[ii],self.batch_size)
            number_sample_slice = len(sample_ind)
            set_ind.append([ii] * number_sample_slice)

            self.set_sample_szie.append(number_sample_slice)
            self.set_index_count[ii] = number_sample_slice

        self.set_ind_seq = np.concatenate(set_ind, axis=0)
        self.set_ind_size = self.set_ind_seq.shape[0]
        self.index_count = self.set_ind_size
        
        # ind = np.random.choice(self.set_ind_size, self.set_ind_size, replace=False)
        # self.set_ind_seq = self.set_ind_seq[ind]        

    def perfdata(self):
        self.xdatfile = self.paras.xdatfile
        self.eigfile  = self.paras.eigfile
        self.kpfile   = self.paras.kpfile

        self.refdir = self.paras.refdir
        self.perfkpionts = np.load(self.refdir + '/' + self.kpfile)
        self.perfeigs = np.load(self.refdir + '/' + self.eigfile)
        if len(self.perfeigs.shape)==3:
            nsp, nk, nb = self.perfeigs.shape
            assert nsp == 1
            self.perfeigs = np.reshape(self.perfeigs,[nk,nb])
        elif len(self.perfeigs.shape)==2:
            nk, nb = self.perfeigs.shape
        else:
            print('The shape of ref eigs is no corrtect.')
            exit()
            
        asetrajs = Trajectory(filename=self.refdir + '/' + self.xdatfile, mode='r')
        self.perfstruct = asetrajs[0]

    def testdata (self):
        # testing data  has lables
        self.xdatfile = self.paras.xdatfile
        self.eigfile  = self.paras.eigfile
        self.kpfile   = self.paras.kpfile
        
        self.testdir   = self.paras.testdir
        self.batch_size  = self.paras.batch_size
        print('# Finding datadir for testings:')
        print('#    - ' + self.testdir)
        asetrajs = Trajectory(filename=self.testdir + '/' + self.xdatfile, mode='r')
        kpionts = np.load(self.testdir + '/' + self.kpfile)
        eigs   = np.load(self.testdir + '/' + self.eigfile)
        # data info. out to log.
        print('#    - ' + self.testdir + ': total snaps' + f'{len(asetrajs):6d}')
        assert len(eigs) == len(asetrajs)
        return asetrajs, eigs, kpionts


    def predictparas(self):
        # prediction data no lables.
        self.inputdir = self.paras.predictdir

    def SampleSlice(self, traj, slice):
        numsampes = len(traj)
        stind=slice[0]
        edind=slice[1]
        assert (stind < edind)
        
        if  stind < numsampes and edind <= numsampes:
            samples  = list(traj[stind:edind])
        elif stind < numsampes and edind > numsampes:
            samples  = list(traj[stind:numsampes])
            samples += list(traj[0:edind-numsampes])
        else:
            samples  = list(traj[stind - numsampes:edind - numsampes])
        return samples

    def traindata(self):
        self.index_count += 1
        if self.index_count > self.set_ind_size:
            self.index_count = 1
            ind = np.random.choice(self.set_ind_size, self.set_ind_size, replace=False)
            # random shuffle the data seq.
            self.set_ind_seq = self.set_ind_seq[ind]

        set_id = self.set_ind_seq[self.index_count-1]

        self.set_index_count[set_id] += 1

        if self.set_index_count[set_id] > self.set_sample_szie[set_id]:
            self.set_index_count[set_id] = 1
            rst = np.random.choice(self.set_dat_sizes[set_id],1,replace=False)[0]
            # sequence with different starting
            self.sliceseq = np.arange(0,self.set_dat_sizes[set_id],self.batch_size) + rst
            self.sliceseq[self.sliceseq >= self.set_dat_sizes[set_id]] -= self.set_dat_sizes[set_id]
            assert(self.set_sample_szie[set_id] == len(self.sliceseq))
            # shuffle
            ind = np.random.choice(
                self.set_sample_szie[set_id], self.set_sample_szie[set_id], replace=False)
            self.sliceseq = self.sliceseq[ind] 

        sampleslice = [self.sliceseq[self.set_index_count[set_id]-1],
                         self.sliceseq[self.set_index_count[set_id]-1] + self.batch_size ]
        
        tr_aseStructs = self.SampleSlice(traj = self.set_asetrajs[set_id], slice=sampleslice)
        
        pkind = np.arange(sampleslice[0],sampleslice[1])
        pkind[pkind >= len(self.seteigs[set_id])] -= len(self.seteigs[set_id])
        #treigs = self.seteigs[set_id][sampleslice[0]:sampleslice[1]]
        treigs = self.seteigs[set_id][pkind]

        valid_id = self.numtrsets - 1
        vldind = np.random.choice(self.set_dat_sizes[valid_id],1,replace=False)[0]
        validslice = [vldind, vldind + self.valid_size]

        valid_aseStructs = self.SampleSlice(traj = self.set_asetrajs[valid_id], slice=validslice)
        
        pkind = np.arange(validslice[0],validslice[1])
        pkind[pkind >= len(self.seteigs[valid_id])] -= len(self.seteigs[valid_id])
        valid_eigs  = self.seteigs[valid_id][pkind]

        return set_id, tr_aseStructs, treigs, valid_id, valid_aseStructs, valid_eigs
        


class EnvBuild(StructBuild):
    """build the local environment for ecah bond.

    inputs
    ------
    paras: the parameters class instance.
    """
    
    def __init__(self,paras):
        super(EnvBuild,self).__init__(paras)
        self.EnvCutOff = paras.EnvCutOff
        self.NumEnv    = paras.NumEnv

    def Projection(self):
        """ Get the envlist of projected each atom  in the whole structure.

        attribute  
        --------- 
        ProjEnv: N*M array, N is the num of projected atoms. M the number of atoms within envcut.
        for each atom. M atoms is the environment, and sorted by atom type id. e.g. C. H system: if  M = 15,   1-10 C 11-15 H

        """
        assert(len(self.ProjAtomType) == len(self.ProjAnglrM))
        self.AtomTypeNOrbs = np.zeros(len(self.ProjAtomType),dtype=int)
        for ii in range(len(self.ProjAtomType)):
            for iorb in self.ProjAnglrM[ii]:
                self.AtomTypeNOrbs[ii] += int(1 + 2 * self.AnglrMID[iorb])

        self.ProjAtoms = np.array([False] * len(self.AtomSymbols))
        for iproj in self.ProjAtomType:
            self.ProjAtoms[np.where(np.asarray(self.AtomSymbols)==iproj)[0]] = True
        symbols_arr = np.array(self.AtomSymbols)

        # define the ase struct Atoms class with only projected atoms.
        self.ProjStruct = Atoms(symbols = symbols_arr[self.ProjAtoms].tolist(), pbc = self.Struct.pbc,
                       cell = self.Struct.cell, positions = self.Struct.positions[self.ProjAtoms])
        
        ilist, jlist, Rlatt = ase.neighborlist.neighbor_list(quantities=['i','j','S'], \
                                    a=self.Struct, cutoff=self.EnvCutOff)
        EnvAllArrs = np.concatenate([np.reshape(ilist,[-1,1]),np.reshape(jlist,[-1,1]),Rlatt],axis=1)
        EnvList    = []
        for ii in range(self.NumAtoms):
            envii = EnvAllArrs[np.where(ilist==ii)[0]]
            EnvList.append(envii[:,1:])
        EnvList  = np.asarray(EnvList,dtype=object)
        ProjEnvList = EnvList[self.ProjAtoms]
        assert(len(ProjEnvList) == np.sum(self.ProjAtoms))
        
        self.tt = ProjEnvList
        ProjEnv = []
        for ii in range(len(ProjEnvList)):
            porjenv = np.asarray(ProjEnvList[ii], dtype=int)
            typeii = np.asarray(self.TypeID)[ porjenv[:,0] ]
            # [typeid, atom_index_j, Rx, Ry, Rz ]
            enviisorted = self.sortarr(typeii, porjenv, dim1 = 1, dim2 = 4)
            ProjEnv.append(enviisorted)
        self.ProjEnv = np.asarray(ProjEnv,dtype=object)
        
        for it in range(len(self.Uniqsybl)):
            numenvit = []
            for ii in range(len(self.ProjEnv)):
                #print(np.sum(self.ProjEnv[ii,:,0] == it ))
                typeii = np.asarray(self.ProjEnv[ii],dtype=int)
                numenvit.append(np.sum(typeii[:,0] == it ))
            if (self.NumEnv[it] <  np.max(numenvit)):
                warnings.warn("NumEnv < np.max(numenvit)")
        # print('# the NumEnv is set to be : ', self.NumEnv)
        
    def iBondEnv(self,ibond):
        """  generate the environment for ecah bond.

        input
        -----
        ibond: i-th bond. ibond = Bonds[i]. has the form 》[i, j, rx,ry, rz]

        return
        ------ 
        envib4: N * 4 array. N is the sum of  NumEnv defined in input.
        """
        numbondenv = np.array(self.NumEnv)
        envib4 = np.zeros([np.sum(numbondenv*2),4])
        ist = 0
        for ib in [0,1]:
            isite = ibond[ib]
            projenvsite = np.asarray(self.ProjEnv[isite],dtype=int)
            envitype   = projenvsite[:,0]
            envlist    = projenvsite[:,1]
            envRfrac   = projenvsite[:,2:]
            envRcart   = np.matmul(envRfrac,self.Lattice)
            isitepos   = self.ProjStruct.positions[isite]
            envi       = self.Positions[envlist] - isitepos +  envRcart
            rr         = np.linalg.norm(envi,axis=1)
            envi_hat   = envi/np.reshape(rr,[-1,1])
            srr        = self.EnvSmoth(rr,rcut=self.EnvCutOff, rcut_smth=self.EnvCutOff*0.8)

            for it in range(len(self.Uniqsybl)):
                if  np.sum(envitype==it) > 0:
                    srrit = np.reshape(srr[envitype==it],[-1,1])
                    envi_hatit = envi_hat[envitype==it]
                    envi_hatit2 = np.concatenate([srrit,envi_hatit],axis=1)
                    envi_hatit2 = np.asarray(sorted(envi_hatit2, key=lambda s:s[0],reverse = True))

                    if np.sum(envitype == it) > numbondenv[it]:
                        print('Warning!, the size of env in cutoff is larger than NumEnv parameter.')
                        ied = ist + numbondenv[it]
                    else:
                        ied = ist + np.sum(envitype==it)
                    #print(ist,ied)
                    envib4[ist:ied]  = envi_hatit2[0:ied-ist]
                ist += numbondenv[it]
        return envib4

    
    @staticmethod
    def EnvSmoth(rr,rcut,rcut_smth):
        srr = np.zeros_like(rr)
        eps = 1.0E-3
        assert((rr-0>eps).all())
        rr_large  = rr[rr >= rcut_smth]
        srr[rr <  rcut_smth] = 1.0 / rr[rr<rcut_smth]
        srr[rr >= rcut_smth] = 1.0 / rr_large * (0.5 * np.cos(np.pi*(rr_large - rcut_smth)/(rcut-rcut_smth)) + 0.5)
        srr[rr >  rcut]      = 0.0
        return srr
        

    @staticmethod
    def sortarr(refarr, tararr, dim1=1, dim2=1, axis=0):
        """ sort the target ndarray using the reference array
        
        inputs
        -----
        refarr: N * M1 reference array
        tararr: N * M2 target array
        dim1: 2-nd dimension  of reference array: ie : M1
        dim2: 2-nd dimension  of target array: ie : M2
        axis: the array is sorted according to the value of refarr[:,axis].

        return
        ------
        sortedarr: sorted array.
        """
        refarr = np.reshape(refarr,[-1,dim1])
        tararr = np.reshape(tararr,[-1,dim2])
        assert(len(refarr)==len(tararr))
        tmparr = np.concatenate([refarr,tararr],axis=1)
        sortedarr = np.asarray(sorted(tmparr,key=lambda s:s[axis]))
        return sortedarr[:,:]
        
    @staticmethod
    def CheckCutOff(idealase, cutoff):
        # use the ideal crystal structure to check the choice of cut off.
        ilist, jlist, Rlatt, dist = ase.neighborlist.neighbor_list(quantities=['i','j','S','d'], \
                            a= idealase, cutoff = cutoff+3)
                            
        Distance=[]
        numatoms = len(idealase.numbers)
        for ii in range(numatoms):
            distii = dist[np.where(ilist==ii)[0]]
            Distance.append(distii)
        # Distance = np.asarray(Distance,dtype=object)
        print('# ' + '--'*18 + ' check cutoff ' + '--'*18)
        print('# reference for choosing the cutoff!')
        print('# distance of 1st to 12th neighbour for each atom: ')
        for ii in range(numatoms):
            distshell = np.sort(np.unique(np.round(Distance[ii],2)))
            minvalind = np.argmin(np.abs(distshell - cutoff))
            print('# atom ', ii+1, ' :  ', distshell[0:minvalind+3])
            if (distshell - cutoff)[minvalind] > 0 and(distshell - cutoff)[minvalind] < 0.05:
                print('# The envcutoff %0.3f within %d to %d - NN' %(cutoff,minvalind-1,minvalind) )
                print('Warning! the envcutoff is too close to next NN. \n' + 
                'This will highly likely introduce more atoms in the atomic environment for some configurations in sampling')
                print('Suggestion: lower envcutoff or increase NumEnv')
            else:
                print('# The envcutoff %0.3f within %d to %d - NN' %(cutoff,minvalind,minvalind+1) )
        
        ilist, jlist, Rlatt, dist = ase.neighborlist.neighbor_list(quantities=['i','j','S','d'], \
                                    a=idealase, cutoff=cutoff)
        Jlist = []
        for ii in range(numatoms):
            jlistii = jlist[np.where(ilist==ii)[0]]
            
            Jlist.append(jlistii)
        #Jlist = np.asarray(Jlist,dtype=object)

        AtomSymbols = idealase.get_chemical_symbols()
        Uniqsybl = [AtomSymbols[0]]
        for i in AtomSymbols:
            if not (i in Uniqsybl):
                Uniqsybl.append(i)
        Type2ID = {}
        for i in range(len(Uniqsybl)):
            Type2ID[Uniqsybl[i]] = i
        TypeID = []
        for i in range(len(AtomSymbols)):
            TypeID.append(Type2ID[AtomSymbols[i]])
        TypeID = np.asarray(TypeID)
        print('# ' + '--'*17 + ' Num atoms in Env ' + '--'*17)
        print('# The nu. of atoms in Env for each type is : at cutoff = %0.3f.' %cutoff)
        print('# pls set NumEnv larger than this.')
        for ii in range(numatoms):
            numenvii = []
            for it in range(len(Uniqsybl)):
                numenvii.append(np.sum(TypeID[Jlist[ii]]==it))
            print('# atom ', ii+1, ' :  ', numenvii)
        print('# ' + '--'*43)