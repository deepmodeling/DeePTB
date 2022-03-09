import numpy as np
import ase

from dpnegf.negf.NEGFStruct import StructNEGFBuild
from dpnegf.nnet.Model import Model
# from negf.RegionHamil import Device_Hamils, Contact_Hamils

class NEGFHamiltonian(object):

    def __init__(self,paras, StructAse):
        self.paras = paras
        self.TBmodel = paras.TBmodel

        self.ngstr = StructNEGFBuild(paras,StructAse)
        self.ngstr.BuildRegions()
        self.mdl = Model(paras)
        self.mdl.structinput(self.ngstr.GeoStruct)
        self.kmesh = paras.kmesh
        self.Contacts = paras.Contacts

    def Scat_Hamils(self):
        if self.TBmodel.lower() == 'nntb':
            self.mdl.loadmodel()
            self.mdl.nnhoppings()
            self.mdl.SKhoppings()
            self.mdl.SKcorrection()
            self.hoppings = self.mdl.hoppings_corr
            self.overlap = self.mdl.overlaps_corr
            self.onsiteEs = self.mdl.onsiteEs_corr
            self.onsiteSs = self.mdl.onsiteSs_corr
        elif self.TBmodel.lower() == 'sktb':
            self.mdl.SKhoppings()
            self.hoppings = self.mdl.skhoppings
            self.overlap = self.mdl.skoverlaps
            self.onsiteEs = self.mdl.onsiteEs
            self.onsiteSs = self.mdl.onsiteSs

        self.mdl.HSmat(hoppings = self.hoppings, overlaps = self.overlap , 
                                    onsiteEs = self.onsiteEs, onsiteSs = self.onsiteSs)
        
        Device_BondHBlock = [ ii.detach().numpy() for ii in self.mdl.BondHBlock]
        Device_BondSBlock = [ ii.detach().numpy() for ii in self.mdl.BondSBlock]

        devH = Device_Hamils(negf_struct_ins=self.ngstr,
            bond_list_ins=self.mdl.bondbuild,hamil_block=Device_BondHBlock,overlap_block=Device_BondSBlock)
        
        Hss, Sss, Bondss, ANorbs_Scat = devH.Gen_Scat_HR()
        kpoints = self.MPSampling(size=self.kmesh)
        ScatDict={} 
        ScatDict['klist'] = kpoints
        norbss = np.sum(ANorbs_Scat)
        nk =len(kpoints)
        ScatDict['Hss'] = np.zeros([nk, norbss, norbss],dtype=complex)
        ScatDict['Sss'] = np.zeros([nk, norbss, norbss],dtype=complex)
        for ikp in range(len(kpoints)):
            kp = kpoints[ikp]
            ScatDict['Hss'][ikp] = self.Hamr2Hamk(hij = Hss, kpoint = kp, \
                            bond = Bondss, orbsi = ANorbs_Scat, orbsj = ANorbs_Scat, conjtran = True)

            ScatDict['Sss'][ikp] = self.Hamr2Hamk(hij = Sss,kpoint = kp, \
                            bond = Bondss, orbsi = ANorbs_Scat, orbsj = ANorbs_Scat, conjtran = True)

        ScatContDict={}
        for itag in self.Contacts:
            ScatContDict[itag] = {}
            Hsc, Ssc, Bondsc, ANorb_s, ANorbs_c = devH.Gen_Cont_Scat_HR(tag=itag)
            norbss = np.sum(ANorbs_Scat)
            norbsc = np.sum(ANorbs_c)
            assert norbss == np.sum(ANorbs_Scat)
            ScatContDict[itag]['Hsc'] = np.zeros([nk, norbss, norbsc],dtype=complex)
            ScatContDict[itag]['Ssc'] = np.zeros([nk, norbss, norbsc],dtype=complex)
            for ikp in range(len(kpoints)):
                kp = kpoints[ikp]
                ScatContDict[itag]['Hsc'][ikp] = self.Hamr2Hamk(hij = Hsc, kpoint = kp, \
                            bond = Bondsc, orbsi = ANorb_s, orbsj = ANorbs_c, conjtran = False)

                ScatContDict[itag]['Ssc'][ikp] = self.Hamr2Hamk(hij = Ssc, kpoint = kp, \
                            bond = Bondsc, orbsi = ANorb_s, orbsj = ANorbs_c, conjtran = False)

        return ScatDict, ScatContDict

    def Cont_Hamils_dev(self):
        if self.TBmodel.lower() == 'nntb':
            self.mdl.loadmodel()
            self.mdl.nnhoppings()
            self.mdl.SKhoppings()
            self.mdl.SKcorrection()
            self.hoppings = self.mdl.hoppings_corr
            self.overlap = self.mdl.overlaps_corr
            self.onsiteEs = self.mdl.onsiteEs_corr
            self.onsiteSs = self.mdl.onsiteSs_corr
        elif self.TBmodel.lower() == 'sktb':
            self.mdl.SKhoppings()
            self.hoppings = self.mdl.skhoppings
            self.overlap = self.mdl.skoverlaps
            self.onsiteEs = self.mdl.onsiteEs
            self.onsiteSs = self.mdl.onsiteSs

        self.mdl.HSmat(hoppings = self.hoppings, overlaps = self.overlap , 
                                    onsiteEs = self.onsiteEs, onsiteSs = self.onsiteSs)
        
        Device_BondHBlock = [ ii.detach().numpy() for ii in self.mdl.BondHBlock]
        Device_BondSBlock = [ ii.detach().numpy() for ii in self.mdl.BondSBlock]

        devH = Device_Hamils(negf_struct_ins=self.ngstr,
            bond_list_ins=self.mdl.bondbuild,hamil_block=Device_BondHBlock,overlap_block=Device_BondSBlock)
        kpoints = self.MPSampling(size=self.kmesh)
        ContDict={}
        ContDict['klist'] = kpoints
        for it in range(len(self.Contacts)):
            tag = self.Contacts[it]
            ContDict[tag] = {}
            H11, H12, S11, S12, bond11, bond12, AtomNorbs = devH.Gen_Cont_HR(tag=tag)
        
            norbs = np.sum(AtomNorbs)
            nk = len(kpoints)
            ContDict[tag]['H00'] = np.zeros([nk,norbs,norbs],dtype=complex)
            ContDict[tag]['H01'] = np.zeros([nk,norbs,norbs],dtype=complex)
            ContDict[tag]['S00'] = np.zeros([nk,norbs,norbs],dtype=complex)
            ContDict[tag]['S01'] = np.zeros([nk,norbs,norbs],dtype=complex)
            
            for ikp in range(len(kpoints)):
                kp = kpoints[ikp]
                ContDict[tag]['H00'][ikp] = self.Hamr2Hamk(hij = H11, kpoint = kp,\
                        bond = bond11, orbsi = AtomNorbs, orbsj = AtomNorbs, conjtran = True)
                ContDict[tag]['H01'][ikp] = self.Hamr2Hamk(hij = H12, kpoint = kp, \
                        bond = bond12, orbsi = AtomNorbs, orbsj = AtomNorbs, conjtran = False)
                ContDict[tag]['S00'][ikp] = self.Hamr2Hamk(hij = S11, kpoint = kp, \
                        bond = bond11, orbsi = AtomNorbs, orbsj = AtomNorbs, conjtran = True)
                ContDict[tag]['S01'][ikp] = self.Hamr2Hamk(hij = S12, kpoint = kp, \
                        bond = bond12, orbsi = AtomNorbs, orbsj = AtomNorbs, conjtran = False)
        
        return ContDict

    def Cont_Hamils(self):
        ContH = Contact_Hamils(negf_struct_ins = self.ngstr)
        kpoints = self.MPSampling(size=self.kmesh)

        ContDict={}
        ContDict['klist'] = kpoints
        for it in range(len(self.Contacts)):
            tag = self.Contacts[it]
            ContDict[tag] = {}
            cont_struct = self.ngstr.ContactStruct[tag]
            self.mdl.structinput(cont_struct,TRsymm=False)
            
            if self.TBmodel.lower() == 'nntb':
                self.mdl.loadmodel()
                self.mdl.nnhoppings()
                self.mdl.SKhoppings()
                self.mdl.SKcorrection()
                self.hoppings = self.mdl.hoppings_corr
                self.overlap = self.mdl.overlaps_corr
                self.onsiteEs = self.mdl.onsiteEs_corr
                self.onsiteSs = self.mdl.onsiteSs_corr

            elif self.TBmodel.lower() == 'sktb':
                self.mdl.SKhoppings()
                self.hoppings = self.mdl.skhoppings
                self.overlap = self.mdl.skoverlaps
                self.onsiteEs = self.mdl.onsiteEs
                self.onsiteSs = self.mdl.onsiteSs


            self.mdl.HSmat(hoppings = self.hoppings, overlaps = self.overlap , 
                                    onsiteEs = self.onsiteEs, onsiteSs = self.onsiteSs)
            
            Cont_BondHBlock = [ ii.detach().numpy() for ii in self.mdl.BondHBlock]
            Cont_BondSBlock = [ ii.detach().numpy() for ii in self.mdl.BondSBlock]

            H00_R, H01_R, S00_R, S01_R, Bond00, Bond01, ANOrb_Cont = ContH.Gen_Cont_HR(
                    bond_list_ins = self.mdl.bondbuild, hamil_block = Cont_BondHBlock, overlap_block=Cont_BondSBlock, contag = tag)
            norbs = np.sum(ANOrb_Cont)
            nk = len(kpoints)
            ContDict[tag]['H00'] = np.zeros([nk,norbs,norbs],dtype=complex)
            ContDict[tag]['H01'] = np.zeros([nk,norbs,norbs],dtype=complex)
            ContDict[tag]['S00'] = np.zeros([nk,norbs,norbs],dtype=complex)
            ContDict[tag]['S01'] = np.zeros([nk,norbs,norbs],dtype=complex)

            for ikp in range(len(kpoints)):
                kp = kpoints[ikp]
                # Trans to k space. 
                ContDict[tag]['H00'][ikp] = self.Hamr2Hamk(hij = H00_R, kpoint = kp,\
                        bond = Bond00, orbsi = ANOrb_Cont, orbsj = ANOrb_Cont, conjtran = False)
                ContDict[tag]['H01'][ikp] = self.Hamr2Hamk(hij = H01_R, kpoint = kp, \
                        bond = Bond01, orbsi = ANOrb_Cont, orbsj = ANOrb_Cont, conjtran = False)
                ContDict[tag]['S00'][ikp] = self.Hamr2Hamk(hij = S00_R, kpoint = kp, \
                        bond = Bond00, orbsi = ANOrb_Cont, orbsj = ANOrb_Cont, conjtran = False)
                ContDict[tag]['S01'][ikp] = self.Hamr2Hamk(hij = S01_R, kpoint = kp, \
                        bond = Bond01, orbsi = ANOrb_Cont, orbsj = ANOrb_Cont, conjtran = False)
        
        return ContDict



    @staticmethod
    def MPSampling(size = [1,1,1]):
        """Construct a uniform sampling of k-space of given size.
        
        Monkhorst-Pack Method: 
        generate k-points as [Hendrik J. Monkhorst and James D. Pack: Special points for Brillouin-zone integrations, Phys. Rev. B 13, 5188–5192 (1976) doi:10.1103/PhysRevB.13.5188]

        \sum_{i=1,2,3} \frac{2 ni - Ni - 1}{2 Ni } \vec{b}_i
        with ni = 1, 2, 3, ..., Ni and bi's are reciprocal lattice vectors.

        Input
        -----
        size = (N1,N2,N3)

        Return
        ------
        Array of kpoints,  shape [N1*N2*N3,  3]

        e.g. 
            >>> MPSampling([1,1,1]) 
            >>> array([0.0, 0.0, 0.0])
        """

        if len(size)!=3:
            raise ValueError('Illegal size, len(size) should be 3: %s' % list(size))

        if np.less_equal(size, 0).any():
            raise ValueError('Illegal size: %s' % list(size))
        
        # size = list(size)
        # if size[self.Iter_Direct] != 1:
        #    warnings.warn('Warning: the BZ samplig along the iteration direction %d sould always be 1. I change it to 1 for you.' %self.Iter_Direct)
    
        kpts = np.indices(size).transpose((1, 2, 3, 0)).reshape((-1, 3))
        return (kpts + 0.5) / size - 0.5
        
    
    @staticmethod
    def Hamr2Hamk(hij, kpoint, bond, orbsi, orbsj, conjtran = False):
        """ Transfer the Hamiltonian for real space to k space.
        
        Inputs
        ------
        hij: Hamiltonian or overlap blocks.
        kpoint: a single k-point
        bond: bond list
        orbsi: atom norbs for <i>s.
        orbsj: atom norbs for <j>s.

        Returns
        -------
        hk: Hamiltonian in k-space (complex matrix) with dimension [sum(orbsi) , sum(orbsj)].
        """

        k = kpoint
        norbsi  = np.sum(orbsi) 
        norbsj  = np.sum(orbsj) 
        hk = np.zeros([norbsi,norbsj],dtype=complex)
        for ib in range(len(bond)):
            R = bond[ib,2:]
            i = bond[ib,0]
            j = bond[ib,1]
            ist = int(np.sum(orbsi[0:i]))
            ied = int(np.sum(orbsi[0:i+1]))
            jst = int(np.sum(orbsj[0:j]))
            jed = int(np.sum(orbsj[0:j+1]))
            # hk[ist:ied,jst:jed] += hij[ib] * np.exp(-1j * 2 * np.pi* np.dot(k,R))
            # hk[ist:ied,jst:jed] +=  np.array(hij[ib],dtype=float) * np.exp(-1j * 2 * np.pi* np.dot(k,R))

            if conjtran:
                if (R == np.array([0,0,0])).all() and  i == j:
                    hk[ist:ied,jst:jed] +=  0.5 * np.array(hij[ib],dtype=float) * np.exp(-1j * 2 * np.pi* np.dot(k,R))
                else:
                    hk[ist:ied,jst:jed] +=  np.array(hij[ib],dtype=float) * np.exp(-1j * 2 * np.pi* np.dot(k,R))      
            else:
                hk[ist:ied,jst:jed] +=  np.array(hij[ib],dtype=float) * np.exp(-1j * 2 * np.pi* np.dot(k,R))
        
        if conjtran:
            hk = hk + hk.T.conj()
        """
        if conjtran:
            if norbsi != norbsj:
                print('Error with Hamr2Hamk, please check the input. orbsi == orbsj ?')
                exit()
            else:
                hk = hk + np.transpose(np.conjugate(hk))
                for i in range(norbsi):
                    hk[i,i] = hk[i,i] * 0.5
        """
        return hk




class Device_Hamils(object):
    """ calss to obtain the Hamiltonian and overlap for the device.

    Inputs
    ------
    negf_struct_ins: instance of StructNEGFBuild  class defined in negf.NEGFStruct.py
    bond_list_ins: instance of  BondListBuild class defined in sktb.StructData.py.
    hamil_block: the Hamiltonian blocks
    overlap_block: the overlap blocks.

    Attributes
    ----------
    ProjAtoms: Array of bool, for all the atoms in The device stucture, is projected [True], is not False.
    ProjRegions: the range of different regions, reconting in projected structure.
    AtomNOrbs: number of orbitals on each projected atoms.
    Bonds: the bondlist of the whole projected structure.
    RegionRanges: the detailed range of different regions. 
    """
    def __init__(self,
                negf_struct_ins,
                bond_list_ins,
                hamil_block,
                overlap_block):

        self.ProjAtoms = negf_struct_ins.ProjAtoms
        self.ProjRegions = negf_struct_ins.ProjRegions
        self.ContactsNames = negf_struct_ins.ContactsNames
        self.Contacts  = negf_struct_ins.Contacts

        
        self.ContactsExtSign = negf_struct_ins.ContactsExtSign
        self.ExtDirect = negf_struct_ins.ExtDirect


        self.AtomNOrbs = bond_list_ins.AtomNOrbs
        self.Bonds = np.concatenate([bond_list_ins.BondsOnSite,bond_list_ins.Bonds],axis=0)
        self.NBonds = self.Bonds.shape[0]

        self.ibond = self.Bonds[:,0]
        self.jbond = self.Bonds[:,1] 

        self.Hamil_Block = hamil_block
        self.Overlap_Block = overlap_block

        # Get more detailed range of different regions.
        # to be more specific, the region of each principal layer (PL).
        self.RegionRanges={}
        for itag in list(self.Contacts):
            atomregion = np.array(self.ProjRegions[itag])-1
            natoms = atomregion[1] - atomregion[0] + 1
            assert(natoms%2==0)
            natomspl  = natoms//2
            PL1Rg = [atomregion[0],atomregion[0] + natomspl-1]
            PL2Rg = [atomregion[0] + natomspl,atomregion[1]]
            self.RegionRanges[itag] = [PL1Rg,PL2Rg]

        itag = 'Device'
        atomregion = np.array(self.ProjRegions[itag])-1
        self.RegionRanges[itag] = [atomregion[0],atomregion[1]]


    def Gen_Cont_Scat_HR(self,tag='Source'):
        """ get the Cont_Scat Hamil.
        
        Inputs
        ------
        tag: the name of contacts, default Source
        
        Return
        ------
        Hsc: Scatter and Contact interaction Hamiltonian.
        Ssc: Basis overlap of Scatter and Contact Hamiltonian.
        Bonds_Scat_Cont: Bond lists of the Scatter and Contact interaction Hamiltonian
        AtomNorbs_Scat: the number of orbitals for the atoms in Scatter regions. 
        AtomNorbs_Cont: the number of orbitals for the atoms in the contact tag.
        """

        assert(tag.lower() in self.ContactsNames)
        Scatter_Rg = self.RegionRanges['Device']
        PL1Rg  = self.RegionRanges[tag][0]  
        Hamil_Scat_Cont_Index = self.Get_Block_Ind(self.ibond, self.jbond, Scatter_Rg, PL1Rg)
        Bonds_Scat_Cont = self.Get_Block_Bond(self.Bonds, Hamil_Scat_Cont_Index, Scatter_Rg, PL1Rg)
        AtomNorbs_Scat = self.AtomNOrbs[Scatter_Rg[0]:Scatter_Rg[1]+1]
        AtomNorbs_Cont = self.AtomNOrbs[PL1Rg[0]:PL1Rg[1]+1]

        #Hscind, bondsc,AtomNorbsS, AtomNorbsC= self.GetScatterHSparas(tag)
        Hsc = np.array(self.Hamil_Block,dtype=object)[Hamil_Scat_Cont_Index]  * 13.605662285137 *2
        Ssc = np.array(self.Overlap_Block,dtype=object)[Hamil_Scat_Cont_Index]
        return Hsc, Ssc, Bonds_Scat_Cont, AtomNorbs_Scat, AtomNorbs_Cont


    def Gen_Scat_HR(self):
        """ get the Scat_Scat Bond/Hamil index, Bonds, AtomNorbs
                
        Return
        ------
        Hss: Scatter and Scatter interaction Hamiltonian.
        Sss: Basis overlap of Scatter and Scatter Hamiltonian.
        Bonds_Scat_Scat: Bond lists of the Scatter and Scatter interaction Hamiltonian
        AtomNorbs_Scat: the number of orbitals for the atoms in Scatter regions. 
        """   

        Scatter_Rg = self.RegionRanges['Device']
        Hamil_Scat_Scat_Index = self.Get_Block_Ind(self.ibond, self.jbond,Scatter_Rg,Scatter_Rg)
        Bonds_Scat_Scat = self.Get_Block_Bond(self.Bonds, Hamil_Scat_Scat_Index, Scatter_Rg,Scatter_Rg)
        AtomNorbs_Scat = self.AtomNOrbs[Scatter_Rg[0]:Scatter_Rg[1]+1]

        Hss = np.array(self.Hamil_Block, dtype=object)[Hamil_Scat_Scat_Index]  * 13.605662285137 * 2
        Sss = np.array(self.Overlap_Block, dtype=object)[Hamil_Scat_Scat_Index]     
        return Hss, Sss, Bonds_Scat_Scat, AtomNorbs_Scat  


    def Gen_Cont_HR(self,tag='Source'):
        """ get the Cont_Scat Hamil.
        
        Inputs
        ------
        tag: the name of contacts, default Source
        
        Return
        ------
        Hsc: Scatter and Contact interaction Hamiltonian.
        Ssc: Basis overlap of Scatter and Contact Hamiltonian.
        Bonds_Scat_Cont: Bond lists of the Scatter and Contact interaction Hamiltonian
        AtomNorbs_Scat: the number of orbitals for the atoms in Scatter regions. 
        AtomNorbs_Cont: the number of orbitals for the atoms in the contact tag.
        """   
        assert(tag.lower() in self.ContactsNames)
        PL1Rg = self.RegionRanges[tag][0]  
        PL2Rg = self.RegionRanges[tag][1]

        H11ind = self.Get_Block_Ind(self.ibond, self.jbond, PL1Rg, PL1Rg)
        H12ind = self.Get_Block_Ind(self.ibond, self.jbond, PL1Rg, PL2Rg)

        bond11 = self.Get_Block_Bond(self.Bonds,H11ind,PL1Rg,PL1Rg)
        bond12 = self.Get_Block_Bond(self.Bonds,H12ind,PL1Rg,PL2Rg)
        bond12[:, 2 + self.ExtDirect[tag]] = self.ContactsExtSign[tag]        

        AtomNorbs = self.AtomNOrbs[PL1Rg[0]:PL1Rg[1]+1]

        H11 = np.array(self.Hamil_Block, dtype=object)[H11ind]  * 13.605662285137 *2
        H12 = np.array(self.Hamil_Block, dtype=object)[H12ind]  * 13.605662285137 *2
        S11 = np.array(self.Overlap_Block, dtype=object)[H11ind]
        S12 = np.array(self.Overlap_Block, dtype=object)[H12ind]


        return H11, H12, S11, S12, bond11, bond12, AtomNorbs


    """    
    def Get_Scatter_Paras(self):
        "" get the Scat_Scat Bond/Hamil index, Bonds, AtomNorbs
                
        Return
        ------
        Hamil_Scat_Scat_Index: the  index of Scatter and Scatter interaction Hamiltonian.
        Bonds_Scat_Scat: Bond lists of the Scatter and Scatter interaction Hamiltonian
        AtomNorbs_Scat: the number of orbitals for the atoms in Scatter regions. 
        ""
        Scatter_Rg = self.RegionRanges['Device']
        Hamil_Scat_Scat_Index = self.Get_Block_Ind(self.ibond, self.jbond,Scatter_Rg,Scatter_Rg)
        Bonds_Scat_Scat = self.Get_Block_Bond(self.Bonds, Hamil_Scat_Scat_Index, Scatter_Rg,Scatter_Rg)
        AtomNorbs_Scat = self.AtomNOrbs[Scatter_Rg[0]:Scatter_Rg[1]+1]
        return Hamil_Scat_Scat_Index, Bonds_Scat_Scat, AtomNorbs_Scat
    """


    """    
    def Get_Cont_Scat_Int_Paras(self,tag = 'Source'):
        "" get the Cont_Scat Bond/Hamil index, Bonds, AtomNorbs
        
        Inputs
        ------
        tag: the name of contacts, default Source
        
        Return
        ------
        Hamil_Scat_Cont_Index: the  index of Scatter and Contact interaction Hamiltonian.
        Bonds_Scat_Cont: Bond lists of the Scatter and Contact interaction Hamiltonian
        AtomNorbs_Scat: the number of orbitals for the atoms in Scatter regions. 
        AtomNorbs_Cont: the number of orbitals for the atoms in the contact tag.
        ""
        assert(tag.lower() in self.ContactsNames)
        Scatter_Rg = self.RegionRanges['Device']
        PL1Rg  = self.RegionRanges[tag][0]  
        Hamil_Scat_Cont_Index = self.Get_Block_Ind(self.ibond, self.jbond, Scatter_Rg, PL1Rg)
        Bonds_Scat_Cont = self.Get_Block_Bond(self.Bonds, Hamil_Scat_Cont_Index, Scatter_Rg, PL1Rg)
        AtomNorbs_Scat = self.AtomNOrbs[Scatter_Rg[0]:Scatter_Rg[1]+1]
        AtomNorbs_Cont = self.AtomNOrbs[PL1Rg[0]:PL1Rg[1]+1]
        return Hamil_Scat_Cont_Index, Bonds_Scat_Cont, AtomNorbs_Scat, AtomNorbs_Cont
    """


    """ 
        The contact parameters have no use, since we have to calculate and obtain the Hamiltonians 
        for contacts in its bulk stucture.

        def Get_Cont_Paras(self,tag = 'Source'):
        "" generate Hamil for iteration surface green method, h11 h12. see  [Sancho, MP Lopez, et al. 1985].  

        Inputs
        ------
        tag: the name of contacts, default Source
        
        Return
        ------
        H11_Index: the  index of Hamiltonian 11
        H12_Index: the  index of Hamiltonian 12
        Bond11: Bond lists of H11
        Bond12: Bond lists of H12
        AtomNorbs: the number of orbitals for the atoms in the contact tag.
        ""

        assert(tag.lower() in self.ContactsNames)
        PL1Rg = self.RegionRanges[tag][0]  
        PL2Rg = self.RegionRanges[tag][1]

        H11ind = self.Get_Block_Ind(self.ibond, self.jbond, PL1Rg, PL1Rg)
        H12ind = self.Get_Block_Ind(self.ibond, self.jbond, PL1Rg, PL2Rg)

        bond11 = self.Get_Block_Bond(self.Bonds,H11ind,PL1Rg,PL1Rg)
        bond12 = self.Get_Block_Bond(self.Bonds,H12ind,PL1Rg,PL2Rg)
        bond12[:, 2 + self.ExtDirect[tag]] = self.ContactsExtSign[tag]        

        AtomNorbs = self.HSreal.AtomNOrbs[PL1Rg[0]:PL1Rg[1]+1]
        return H11ind, H12ind, bond11, bond12, AtomNorbs"""

    @staticmethod
    def Get_Block_Ind(ibond, jbond, irange, jrange):
        """Getto get the block's index of in bond list. e.g.: the H_{contact}_{scatter} Hamiltonian bolck's index.
        
        Inputs
        ------
        ibond: a list of i for bond <i,j>
        jbond: a list of j for bond <i,j>
        irange: range of i
        jrange: range of j

        Return
        ------
        bondindex: index of the bond that within the range defined by irange and jrange.
        """
        assert len(ibond) == len(jbond)
        ind_arr = np.arange(len(ibond))
        i_ind_tmp1 = ind_arr[ibond >= irange[0]]
        ibond_tmp1 = ibond[ibond >= irange[0]]
        i_ind = i_ind_tmp1[ibond_tmp1<=irange[1]]

        j_ind_tmp1 = ind_arr[jbond >= jrange[0]]
        jbond_tmp1 = jbond[jbond >= jrange[0]]
        j_ind = j_ind_tmp1[jbond_tmp1 <= jrange[1]]
        bondindex = np.intersect1d(i_ind, j_ind)

        return bondindex
    

    @staticmethod
    def Get_Block_Bond(bond,ind,irange,jrange):
        """ using the index to get the bonds.

        Inputs
        ------
        bond: bond list of the whole system
        ind: index of the bond that within the range defined by irange and jrange
        irange: range of i
        jrange: range of j

        Return
        ------
        bonds_in_range: index of the bond that within the range defined by irange and jrange.
        """
        bonds_in_range = bond[ind,:]
        bonds_in_range[:,0] -= irange[0]
        bonds_in_range[:,1] -= jrange[0]
        return bonds_in_range



class Contact_Hamils(object):
    """ calss to obtain the Hamiltonian and overlap for the Contact. 
    
    Note: The contact Hamiltonian must be obtained in bulk structure instead of in the device structure. 

    Inputs
    ------
    negf_struct_ins: instance of StructNEGFBuild  class defined in negf.NEGFStruct.py

    """
    def __init__(self, negf_struct_ins):
        
        self.ContactsNames = negf_struct_ins.ContactsNames
        self.Contacts  = negf_struct_ins.Contacts
        # assert contag.lower() in self.ContactsNames

        self.ContactStruct = negf_struct_ins.ContactStruct
        self.ContactsExtSign = negf_struct_ins.ContactsExtSign
        self.ExtDirect = negf_struct_ins.ExtDirect


    def Gen_Cont_HR(self,
                    bond_list_ins,
                    hamil_block,
                    overlap_block,
                    contag='Source'):
        """Generates contact Hamiltonians, H00 H01
        
        Inputs
        ------
        bond_list_ins: instance of  BondListBuild class defined in sktb.StructData.py.
        hamil_block: the Hamiltonian blocks
        overlap_block: the overlap blocks.
        contag: the contact name. defined paras.Contacts.

        Return
        ------
        H00_Block: Hamiltonian Block with index 00 along the iteration direction.
        H01_Block: Hamiltonian Block with index 01 along the iteration direction.
        S00_Block: overlap Block with index 00 along the iteration direction.
        S01_Block: overlap Block with index 01 along the iteration direction.
        Bond00: bonds list for 00 Block
        Bond01: bonds list for 01 Block
        AtomNOrbs: number of orbitals on each atom in contact.
        """
        assert contag in self.Contacts

        # self.struct = self.ContactStruct[contag]
        ExtSign = self.ContactsExtSign[contag]
        ExtDirection = self.ExtDirect[contag]
        AtomNOrbs = bond_list_ins.AtomNOrbs
        Bonds = np.concatenate([bond_list_ins.BondsOnSite,bond_list_ins.Bonds],axis=0)

        # R01 = 1 or -1. according to angle b/w the iteration direction and the lattice vector.
        R01 = int(ExtSign * 1)
        
        """
        Only the transport direction (also the direction of iteration for surface Green's function) take to be 0 or 1. 
        To get the Hamiltonian Block H00 and H01.
        """

        Rlist = Bonds[:,2:]
        ind_00 = Rlist[:,ExtDirection] == 0
        ind_01 = Rlist[:,ExtDirection] == R01
        # ind_01_p = Rlist[:,ExtDirection] == int(-1*R01)

        Bond00 = Bonds[ind_00]
        Bond01 = Bonds[ind_01]
        
        #Bond01_tmp = Bonds[ind_01_p]
        #Bond01_p  =  np.zeros_like(Bond01_tmp)
        #Bond01_p[:,0] = Bond01_tmp[:,1]
        #Bond01_p[:,1] = Bond01_tmp[:,0]
        #Bond01_p[:,2:] = -1 * Bond01_tmp[:,2:]

        #Bond01 = np.concatenate([Bond01,Bond01_p])

        # H00_Block = hamil_block[ind_00] 
        H00_Block = np.array(hamil_block, dtype=object)[ind_00]  * 13.605662285137 * 2
        # H01_Block = hamil_block[ind_01]
        H01_Block = np.array(hamil_block, dtype=object)[ind_01]  * 13.605662285137 * 2
        #H01_Block_p = np.array(hamil_block, dtype=object)[ind_01_p]  * 13.605662285137 * 2
        #Block_tmp = np.array([ii.T for ii in H01_Block_p])

        #H01_Block = np.concatenate([H01_Block,Block_tmp])

        # S00_Block = overlap_block[ind_00]
        S00_Block = np.array(overlap_block, dtype=object)[ind_00]
        # S01_Block = overlap_block[ind_01]
        S01_Block = np.array(overlap_block, dtype=object)[ind_01]
        #S01_Block_p = np.array(overlap_block, dtype=object)[ind_01_p]

        # Block_tmp = np.array([ii.T for ii in S01_Block_p])
        
        #S01_Block = np.concatenate([S01_Block,Block_tmp])
    

        return H00_Block, H01_Block, S00_Block, S01_Block, Bond00, Bond01, AtomNOrbs
        