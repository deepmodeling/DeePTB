
import numpy as np
#from sktb.SlaterKosterPara import SlaterKosterInt
#from sktb.RotateSK import RotateHS
#from sktb.StructData import StructBuild, BondListBuild
#from sktb.StructData import StructNEGFBuild


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
        contind = self.ContactsNames.index(tag.lower())
        tag = self.Contacts[contind]

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
        assert(contag.lower() in self.ContactsNames)
        contind = self.ContactsNames.index(contag.lower())
        contag = self.Contacts[contind]

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

        Bond00 = Bonds[ind_00]
        Bond01 = Bonds[ind_01]

        # H00_Block = hamil_block[ind_00] 
        H00_Block = np.array(hamil_block, dtype=object)[ind_00]  * 13.605662285137 * 2
        # H01_Block = hamil_block[ind_01]
        H01_Block = np.array(hamil_block, dtype=object)[ind_01]  * 13.605662285137 * 2


        # S00_Block = overlap_block[ind_00]
        S00_Block = np.array(overlap_block, dtype=object)[ind_00]
        # S01_Block = overlap_block[ind_01]
        S01_Block = np.array(overlap_block, dtype=object)[ind_01]


        return H00_Block, H01_Block, S00_Block, S01_Block, Bond00, Bond01, AtomNOrbs
        