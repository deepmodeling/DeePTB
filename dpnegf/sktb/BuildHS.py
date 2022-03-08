
import numpy as np

from sktb.SlaterKosterPara import SlaterKosterInt
from sktb.RotateSK import RotateHS
from sktb.StructData import StructBuild, BondListBuild
from sktb.StructData import StructNEGFBuild

class BuildHSreal(RotateHS):
    """ build hamiltonian and overlap matrix  in real space.

    Attributes:
        au2Ang: atomic unit to \AA. sk para. defined in a.u.
        Bonds: bond list, [..., [i,j], ...]
        BondSBlock: a lists of overlap matrix w.r.t. bond list. [..., [S-mat block], ...]
        BondHBlock: a lists of hamiltonian matrix w.r.t. bond list. [..., [H-mat block], ...]
    """
    def __init__(self,BondBuildIns):
        """init with BondBuild instance.
        
        Args: 
            BondBuildIns: instance of BondBuild class.
        """
        super(BuildHSreal,self).__init__(rot_type='array')

        self.SKAnglrMHSID = {'dd':np.array([0,1,2]), 
                             'dp':np.array([3,4]), 'pd':np.array([3,4]), 
                             'pp':np.array([5,6]), 
                             'ds':np.array([7]),   'sd':np.array([7]),
                             'ps':np.array([8]),   'sp':np.array([8]),
                             'ss':np.array([9])}

        self.au2Ang = 0.529177249 
        self.Bonds = BondBuildIns.Bonds
        self.NBonds = BondBuildIns.NBonds
        self.StructAse = BondBuildIns.Struct
        self.Lattice = BondBuildIns.Lattice
        self.TypeID = BondBuildIns.TypeID
        self.AtomNOrbs = BondBuildIns.AtomNOrbs
        self.AtomTypeNOrbs = BondBuildIns.AtomTypeNOrbs
        self.ProjAnglrM = BondBuildIns.ProjAnglrM
        self.AnglrMID = BondBuildIns.AnglrMID
        self.BondsOnSite = BondBuildIns.BondsOnSite
        self.NumAtoms = BondBuildIns.NumAtoms

    def HSMatBuild(self,SKIntIns):
        """Get hamiltonian and overlap paras. from SlaterKosterInt instance.

        Args: 
            SKIntIns: instance of SlaterKosterInt class.
        """
        self.BondSBlock = []
        self.BondHBlock = []
        if len(self.BondsOnSite) != self.NumAtoms:
            print('Check the num of onsites.')
        for ib in range(len(self.BondsOnSite)):
            ibond  = self.BondsOnSite[ib]
            iatype, jatype = self.TypeID[ibond[0]] , self.TypeID[ibond[1]]
            if iatype != jatype:
                print('Error!, On site bond itype must = j type.')

            Hamilblock = np.zeros([self.AtomTypeNOrbs[iatype], self.AtomTypeNOrbs[jatype]])
            Soverblock = np.zeros([self.AtomTypeNOrbs[iatype], self.AtomTypeNOrbs[jatype]])

            ist = 0    
            for ish in self.ProjAnglrM[iatype]:
                shidi = self.AnglrMID[ish]
                norbi = 2*shidi + 1
                jst = 0
                row,col = np.diag_indices_from(Hamilblock[ist:ist+norbi, ist:ist+norbi])
                row = row + ist
                col = col + ist
                Hamilblock[row,col] = SKIntIns.SiteE[iatype][shidi]
                Soverblock[row,col] = 1.0

                ist = ist +norbi
            self.BondSBlock.append(Soverblock)
            self.BondHBlock.append(Hamilblock)
        #self.BondHsite = np.asarray(BondHsite)
        #self.BondSsite = np.asarray(BondSsite)


        #self.BondSBlock = []
        #self.BondHBlock = []

        for ib in range(self.NBonds):
            ibond = self.Bonds[ib]
            dirvec = self.StructAse.positions[ibond[1]] - self.StructAse.positions[ibond[0]] + \
                np.dot(ibond[2:], self.Lattice)
            dist = np.linalg.norm(dirvec)
            dirvec = dirvec/dist
            dist = dist/self.au2Ang  # to a.u. ; The sk files is written in atomic unit
            iatype, jatype = self.TypeID[ibond[0]] , self.TypeID[ibond[1]]
            # print (ibond, iatype, jatype, dist)

            Hamilblock = np.zeros([self.AtomTypeNOrbs[iatype], self.AtomTypeNOrbs[jatype]])
            Soverblock = np.zeros([self.AtomTypeNOrbs[iatype], self.AtomTypeNOrbs[jatype]])
        
            if iatype == jatype:
                HKinterp12 = SKIntIns.IntpSK(itype=iatype,jtype=jatype,dist=dist)
                # view, the same addr. in mem.
                HKinterp21 = HKinterp12
            else:
                HKinterp12 = SKIntIns.IntpSK(itype=iatype,jtype=jatype,dist=dist)
                HKinterp21 = SKIntIns.IntpSK(itype=jatype,jtype=iatype,dist=dist)

            ist = 0    
            # ind = 0
            for ish in self.ProjAnglrM[iatype]:
                shidi = self.AnglrMID[ish]
                norbi = 2*shidi+1
                jst = 0

                for jsh in self.ProjAnglrM[jatype]:
                    shidj = self.AnglrMID[jsh]
                    norbj = 2 * shidj + 1

                    if shidi < shidj:
                        Hvaltmp = HKinterp12[self.SKAnglrMHSID[ish+jsh]]
                        Svaltmp = HKinterp12[self.SKAnglrMHSID[ish+jsh] +10]

                        tmpH = self.RotHS(Htype=ish+jsh, Hvalue = Hvaltmp, Angvec = dirvec)
                        Hamilblock[ist:ist+norbi, jst:jst+norbj] = np.transpose(tmpH)

                        tmpS = self.RotHS(Htype=ish+jsh, Hvalue = Svaltmp, Angvec = dirvec)
                        Soverblock[ist:ist+norbi, jst:jst+norbj] = np.transpose(tmpS)
                    else:               
                        Hvaltmp = HKinterp21[self.SKAnglrMHSID[ish+jsh]]
                        Svaltmp = HKinterp21[self.SKAnglrMHSID[ish+jsh] +10]

                        # (-1.0)**(shidi + shidj) i>j j>i with the same sign.
                        tmpH = self.RotHS(Htype=jsh+ish, Hvalue = Hvaltmp, Angvec = dirvec)
                        Hamilblock[ist:ist+norbi, jst:jst+norbj] = (-1.0)**(shidi + shidj) * tmpH

                        tmpS = self.RotHS(Htype=jsh+ish, Hvalue = Svaltmp, Angvec = dirvec)
                        Soverblock[ist:ist+norbi, jst:jst+norbj] = (-1.0)**(shidi + shidj) * tmpS                

                    jst = jst + norbj    

                ist = ist + norbi
            
            #print( iatype, jatype, Soverblock.shape)
            self.BondSBlock.append(Soverblock)
            self.BondHBlock.append(Hamilblock)
        
        self.Bonds = np.concatenate([self.BondsOnSite,self.Bonds],axis=0)
        self.NBonds = self.Bonds.shape[0]
    

class DeviceHS(StructNEGFBuild):
    """ build hamiltonian and overlap matrix  in real space for NEGF device.

    Attributes:
        GenContHamilReal: function to gen H and S of contact region.
        GenScatterHamilReal: function to gen H and S of scatter region.
        GenDeviceHamilReal: function to gen H and S of device region.
    """

    def __init__(self,paras) -> None:
        """init DeviceHS class using paras instance.

        Args: 
            paras: instance of Paras class.
        """
        print('# init DeviceHS calss.')
        super(DeviceHS,self).__init__(paras)
        self.BuildRegions()
        bondlst = BondListBuild(paras)
        bondlst.BondStuct(self.GeoStruct)
        bondlst.GetBonds(CutOff=self.CutOff)
        self.ProjAtoms = bondlst.ProjAtoms
        SKint = SlaterKosterInt(paras)
        SKint.ReadSKfiles()
        SKint.IntpSKfunc()
        self.HSreal = BuildHSreal(bondlst)
        self.HSreal.HSMatBuild(SKint)
        self.ibond = self.HSreal.Bonds[:,0]
        self.jbond = self.HSreal.Bonds[:,1] 
    
    def GenContHamilReal(self,tag='Source'):
        """Contact hamiltonian.
        
        Args: 
            tag: "Source" or "Drain" indicate  contact
        """
        assert(tag.lower() in self.ContactsNames)
        H11ind, H12ind, bond11, bond12, AtomNorbs = self.GetContactHSparas(tag)
        H11 = np.array(self.HSreal.BondHBlock,dtype=object)[H11ind]  * 13.605662285137 *2
        H12 = np.array(self.HSreal.BondHBlock,dtype=object)[H12ind]  * 13.605662285137 *2
        S11 = np.array(self.HSreal.BondSBlock,dtype=object)[H11ind]
        S12 = np.array(self.HSreal.BondSBlock,dtype=object)[H12ind]

        return H11, H12, S11, S12, bond11, bond12, AtomNorbs

    def GenScatterHamilReal(self,tag='Source'):
        assert(tag.lower() in self.ContactsNames)
        Hscind, bondsc,AtomNorbsS, AtomNorbsC= self.GetScatterHSparas(tag)
        Hsc = np.array(self.HSreal.BondHBlock,dtype=object)[Hscind]  * 13.605662285137 *2
        Ssc = np.array(self.HSreal.BondSBlock,dtype=object)[Hscind]
        return Hsc, Ssc, bondsc, AtomNorbsS, AtomNorbsC
    
    def GenDeviceHamilReal(self):
        Hssind, bondss, AtomNorbs = self.GetDeviceHSparas()
        Hss = np.array(self.HSreal.BondHBlock,dtype=object)[Hssind]  * 13.605662285137 * 2
        Sss = np.array(self.HSreal.BondSBlock,dtype=object)[Hssind]     
        return Hss, Sss, bondss, AtomNorbs  

    def GetContactHSparas(self,tag = 'Source'):
        assert(tag.lower() in self.ContactsNames)
        if not hasattr(self,'RegionRanges'):
            self.GetRegionsRanges()
        # for iteration surface green method, h11 h12. see  [Sancho, MP Lopez, et al. 1985].  
        PL1Rg = self.RegionRanges[tag][0]  
        PL2Rg = self.RegionRanges[tag][1]

        H11ind = self.GetBlockiInd(self.ibond, self.jbond, PL1Rg, PL1Rg)
        H12ind = self.GetBlockiInd(self.ibond, self.jbond, PL1Rg, PL2Rg)

        bond11 = self.GetBlockBond(self.HSreal.Bonds,H11ind,PL1Rg,PL1Rg)
        bond12 = self.GetBlockBond(self.HSreal.Bonds,H12ind,PL1Rg,PL2Rg)
        bond12[:, 2 + self.ExtDirect[tag]] = self.ContactsExtSign[tag]        

        AtomNorbs = self.HSreal.AtomNOrbs[PL1Rg[0]:PL1Rg[1]+1]
        return H11ind, H12ind, bond11, bond12, AtomNorbs

    def GetDeviceHSparas(self):
        if not hasattr(self,'RegionRanges'):
            self.GetRegionsRanges()
        devsatoms = self.RegionRanges['Device']
        Hssind = self.GetBlockiInd(self.ibond, self.jbond,devsatoms,devsatoms)
        bondss   = self.GetBlockBond(self.HSreal.Bonds,Hssind,devsatoms,devsatoms)
        AtomNorbs = self.HSreal.AtomNOrbs[devsatoms[0]:devsatoms[1]+1]
        return Hssind, bondss, AtomNorbs


    def GetScatterHSparas(self,tag = 'Source'):
        assert(tag.lower() in self.ContactsNames)
        if not hasattr(self,'RegionRanges'):
            self.GetRegionsRanges()
        devsatoms = self.RegionRanges['Device']
        PL1Rg  = self.RegionRanges[tag][0]  
        Hscind = self.GetBlockiInd(self.ibond, self.jbond, devsatoms, PL1Rg)
        bondsc = self.GetBlockBond(self.HSreal.Bonds,Hscind,devsatoms,PL1Rg)
        AtomNorbsS = self.HSreal.AtomNOrbs[devsatoms[0]:devsatoms[1]+1]
        AtomNorbsC = self.HSreal.AtomNOrbs[PL1Rg[0]:PL1Rg[1]+1]
        return Hscind, bondsc, AtomNorbsS, AtomNorbsC


    def GetRegionsRanges(self):
        if not hasattr(self,'ProjRegions'):
            self.GetProjRegions()

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

    def GetProjRegions(self):
        self.ProjRegions={}
        ist = 0
        ied = 0
        for itag in ['Device'] + list(self.Contacts):
            ist = ied +1
            OriRegion = np.array(self.GeoRegions[itag]) - 1 
            projind = self.ProjAtoms[OriRegion[0]:OriRegion[1]+1]
            ied  = np.sum(projind) + ist - 1
            self.ProjRegions[itag] = [ist,ied]
        
    @staticmethod
    def GetBlockiInd(ibond, jbond, irange, jrange):
        if len(ibond) != len(jbond):
            print('Error!')
            exit()
        ind_arr = np.arange(len(ibond))

        i_ind_tmp1 = ind_arr[ibond >= irange[0]]
        ibond_tmp1 = ibond[ibond >= irange[0]]
        i_ind = i_ind_tmp1[ibond_tmp1<=irange[1]]

        j_ind_tmp1 = ind_arr[jbond >= jrange[0]]
        jbond_tmp1 = jbond[jbond >= jrange[0]]
        j_ind = j_ind_tmp1[jbond_tmp1<=jrange[1]]
        h11_ind = np.intersect1d(i_ind,  j_ind)
        return h11_ind
    
    @staticmethod
    def GetBlockBond(bond,ind,range1,range2):
        bond_block = bond[ind,:]
        bond_block[:,0]-=range1[0]
        bond_block[:,1]-=range2[0]
        return bond_block
