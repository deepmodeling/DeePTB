import numpy as np
import sys
import ase 
import ase.io
import ase.neighborlist
from ase import Atoms


class StructBuild(object):
    def __init__(self,paras):
        print('# init StructBuild class.')
        self.atomic_num_dict = ase.atom.atomic_numbers
        self.atomic_num_dict_r =  dict(zip(self.atomic_num_dict.values(), self.atomic_num_dict.keys()))
        self.AnglrMID = {'s':0,'p':1,'d':2,'f':3}
        # sort the atoms types.
        self.AtomType = self.get_uniq_symbol(paras.AtomType)
        self.ProjAtomType = self.get_uniq_symbol(paras.ProjAtomType)
        # self.ProjAnglrM   = paras.ProjAnglrM
        # arange the proj orbitals acoring to the sorted projtype
        self.ProjAnglrM = []
        self.ProjValElec = []
        for ii in self.ProjAtomType:
            self.ProjAnglrM.append(paras.ProjAnglrM[ii])
            self.ProjValElec.append(paras.ValElec[ii])
        self.CutOff = paras.CutOff
        
    def IniUsingFile(self,StructFileName,StructFileFmt):
        struct = ase.io.read(filename= StructFileName, format=StructFileFmt)
        self.IniUsingAse(struct)
        
    def IniUsingAse(self,StructAse):
        self.Struct = StructAse
        self.AtomSymbols = StructAse.get_chemical_symbols()
        self.NumAtoms  = len(self.AtomSymbols)
        self.Positions = StructAse.positions
        self.Lattice   =  np.array(StructAse.cell)
        
        self.Uniqsybl = self.get_uniq_symbol(atomsymbols = self.AtomSymbols)
        # [self.AtomSymbols[0]]

        #for i in self.AtomSymbols:
        #    if not (i in self.Uniqsybl):
        #        self.Uniqsybl.append(i)
            
        self.Type2ID = {}
        for i in range(len(self.Uniqsybl)):
            self.Type2ID[self.Uniqsybl[i]] = i
        self.TypeID = []
        for i in range(self.NumAtoms):
            self.TypeID.append(self.Type2ID[self.AtomSymbols[i]])


    def get_uniq_symbol(self, atomsymbols):
       atom_num=[]
       for it in atomsymbols:
           atom_num.append(self.atomic_num_dict[it])
       # uniq and sort.
       uniq_atom_num = sorted(np.unique(atom_num),reverse=True)
       # assert(len(uniq_atom_num) == len(atomsymbols))
       atomtype = []
       for ia in uniq_atom_num:
           atomtype.append(self.atomic_num_dict_r[ia])

       return atomtype

    def Projection(self):
        # atom_symbols_arr = np.array(atom_symbols)
        assert(len(self.ProjAtomType) == len(self.ProjAnglrM))
        self.AtomTypeNOrbs = np.zeros(len(self.ProjAtomType),dtype=int)
        for ii in range(len(self.ProjAtomType)):
            for iorb in self.ProjAnglrM[ii]:
                self.AtomTypeNOrbs[ii] += int(1 + 2 * self.AnglrMID[iorb])
                
        #projind = []
        self.ProjAtoms = np.array([False] * len(self.AtomSymbols))
        for iproj in self.ProjAtomType:
            self.ProjAtoms[np.where(np.asarray(self.AtomSymbols)==iproj)[0]] = True
            # ind_tmp = np.where(np.asarray(self.AtomSymbols)==iproj)[0]
            #projind.append(ind_tmp.tolist())
        #projind = np.concatenate(projind)
        symbols_arr = np.array(self.AtomSymbols)

        #self.ProjStruct = Atoms(symbols = symbols_arr[projind].tolist(), pbc = self.Struct.pbc,
        #               cell = self.Struct.cell, positions = self.Struct.positions[projind])   
        self.ProjStruct = Atoms(symbols = symbols_arr[self.ProjAtoms].tolist(), 
                                    pbc = self.Struct.pbc, cell = self.Struct.cell, 
                                    positions = self.Struct.positions[self.ProjAtoms])

        # self.ProjStruct = self.Struct
    
class StructNEGFBuild(StructBuild):
    def __init__(self,paras):
        print('# init StructNEGFBuild calss.')
        super(StructNEGFBuild,self).__init__(paras)
        #StructBuild.__init__(StructBuild)
        self.IniUsingFile(StructFileName = paras.StructFileName,StructFileFmt = paras.StructFileFmt)
        self.DeviceRegion = paras.DeviceRegion
        self.Contacts = paras.Contacts
        self.ContactsRegions = paras.ContactsRegions
        self.PrinLayNunit    = paras.PrinLayNunit
        self.ContactsNames = [item.lower() for item in self.Contacts]
    
    def BuildRegions(self):
        # contacts
        self.ExtDirect       = {} 
        self.ContactStruct   = {}
        self.ContactsExtSign = {}
        for itag in self.Contacts:
            ExtDirection, Rsign, ContactStruct = self.BuildContact(tag = itag)
            self.ContactsExtSign[itag] = Rsign
            self.ContactStruct[itag]   = ContactStruct
            self.ExtDirect[itag]       = ExtDirection
            # Device and scatter regions.
        self.GeoRegions, self.GeoStruct = self.BuildDevice(tag='All')
        ase.io.write('OutDeviceStr.extxyz',self.GeoStruct,format='extxyz')
        print('#'+'-'*60)
        
        print('#' + ' print device struct in extxyz format')
        for ik in self.GeoRegions.keys():
            print("# \'"+ik+"\'" + ' atom ranges: ', self.GeoRegions[ik])
        for itag in self.Contacts:
            print('#' + ' print ' + itag + ' struct in extxyz format')
            ase.io.write(itag+'Str.extxyz',self.ContactStruct[itag],format='extxyz')
        print('#'+'-'*60)
        
        
    
    #def BuildGeometry(self):
    #    GeoContRegions, GeoStruct = self.BuildDevice(tag='All')
    #    return GeoContRegions, GeoStruct

    def BuildContact(self,tag='Source'):
        assert(tag.lower() in self.ContactsNames)
        contind = self.ContactsNames.index(tag.lower())
        contname = self.Contacts[contind]
        Atomrange = np.asarray(self.ContactsRegions[contname],dtype=int) - 1
        NatomsCont = (Atomrange[1] - Atomrange[0] + 1)
        NatomsContLayer = NatomsCont//2
        assert(NatomsContLayer%2 ==0)
        postions = self.Struct.positions[Atomrange[0]:Atomrange[1]+1]
        unitlayer1 = postions[0 : NatomsContLayer]
        unitlayer2 = postions[NatomsContLayer: NatomsContLayer*2]
        symbols    = self.Struct.symbols[Atomrange[0]:Atomrange[1]+1]
        unitstmbol = symbols[0 : NatomsContLayer]

        diffv = unitlayer2 - unitlayer1
        assert(np.isclose(diffv[:],diffv[0],atol=1e-06).all())
        ExtendDirect = [not item for item in np.isclose(diffv[0],0)]
        assert(np.sum(ExtendDirect)==1)
        ExtendDirectInd = np.array([0,1,2])[ExtendDirect][0]

        PeriodicCond = [False,False,False]
        Lattice  = np.zeros([3,3])
        for i in range(3):
            if  not ExtendDirect[i]:
                if self.Struct.pbc[i]:
                    PeriodicCond[i] = True
                    Lattice[i] = self.Struct.cell[i]
                else:
                    PeriodicCond[i] = False
                    Lattice[i,i]    = 500
            else:
                PeriodicCond[i] = True
                Lattice[i] = diffv[0]
        Volume = np.dot(Lattice[0], np.cross(Lattice[1],Lattice[2]))
        if Volume < 0:
            Lattice[ExtendDirect] = -1*Lattice[ExtendDirect]
        Volume = np.dot(Lattice[0], np.cross(Lattice[1],Lattice[2]))
        assert(Volume>0)
        Rsign = int(np.sign(np.dot(diffv[0],Lattice[ExtendDirectInd])))
        # Contactstruct = Atoms(symbols=unitstmbol,positions=unitlayer1,
        # pbc=PeriodicCond,cell=Lattice)
        PLpositions = np.copy(unitlayer1)
        PLNunits = self.PrinLayNunit[contname]
        for ii in range(1,PLNunits):
            PLpositions = np.concatenate([PLpositions, 
                    unitlayer1 + ii * Rsign * Lattice[ExtendDirectInd]],axis=0)
        PLsymbols = list(unitstmbol) * PLNunits
        Lattice[ExtendDirectInd,ExtendDirectInd] *= PLNunits

        # construct principal layer using serveral unit cell, 
        # along the iteration direction.
        ContactStruct = Atoms(symbols=PLsymbols, positions=PLpositions, 
                                    pbc=PeriodicCond, cell=Lattice)


        return ExtendDirectInd, Rsign, ContactStruct

    def BuildDevice(self,tag='Scatter'):
        assert(tag.lower() in ['scatter','all'])
        if tag.lower() == 'scatter':
            NPL=1
        else:
            NPL=2
        assert(hasattr(self,'ContactStruct'))

        Atomrange  = np.asarray(self.DeviceRegion,dtype=int) - 1
        NatomsCont = (Atomrange[1] - Atomrange[0] + 1)
        postions   = self.Struct.positions[Atomrange[0]:Atomrange[1]+1]
        symbols    = self.Struct.symbols[Atomrange[0]:Atomrange[1]+1]

        ist = self.DeviceRegion[0]
        ied = self.DeviceRegion[1]
        ScatterSymbols   = list(symbols)
        ScatterLattice   = np.copy(self.Lattice)
        ScatterPositions = np.copy(postions)
        PeriodicCond     = self.Struct.pbc
        PLContactRegions = {'Device':self.DeviceRegion}
        for itag in self.Contacts:
            Rsign     = self.ContactsExtSign[itag]
            ExtDirect = self.ExtDirect[itag]
            contstr   = self.ContactStruct[itag]
            ist = ied +1
            ied = ist + len(contstr.positions)*NPL - 1     
            PLContactRegions[itag] = [ist,ied]
            ScatterSymbols = ScatterSymbols + list(contstr.symbols) * NPL
            for ipl in range(NPL):
                ScatterPositions = np.concatenate([ScatterPositions, 
                    contstr.positions + ipl * Rsign * contstr.cell[ExtDirect]],axis=0)
            ScatterLattice[ExtDirect] = 0.0
            PeriodicCond[ExtDirect]   = False
        ScatterStruct = Atoms(symbols=ScatterSymbols, positions=ScatterPositions,
                                 cell=ScatterLattice, pbc=PeriodicCond)

        return PLContactRegions, ScatterStruct


class BondListBuild(StructBuild):
    def __init__(self, paras):
        print('# init BondListBuild calss.')
        #if not hasattr(self,'ProjStruct'):
        super(BondListBuild,self).__init__(paras)
        #StructBuild.__init__(StructBuild)
    
    def BondStuct(self,struct):
        self.IniUsingAse(struct)
        self.Projection()
        self.IniUsingAse(self.ProjStruct)
        # upload the struct.
        # self.AtomTypeNOrbs = self.AtomTypeNOrbs
        # self.ProjAtoms     = self.ProjAtoms
        # self.CutOff        = self.CutOff
        self.AtomNOrbs = []
        for ii in self.TypeID:
            self.AtomNOrbs.append(self.AtomTypeNOrbs[ii])

        # print(StructBuildIns.atom_norbs)
        
    def GetBonds(self,TRsymm=True, CutOff = -1):
        if not(CutOff>0):
            if hasattr(self,'CutOff'):
                CutOff = self.CutOff    
            else:
                print('Please set CurOff for bond > 0')
                sys.exit()
        
        ilist, jlist, Rlatt = ase.neighborlist.neighbor_list(quantities=['i','j','S'],
                             a=self.Struct, cutoff=CutOff)
        bonds = np.concatenate([np.reshape(ilist,[-1,1]),
                                     np.reshape(jlist,[-1,1]), Rlatt],axis=1)
        
        nbonds = bonds.shape[0]
        if TRsymm:
            bonds_rd = []
            for inb in range(nbonds):
                atomi, atomj, R = bonds[inb,0], bonds[inb,1], bonds[inb,2:]
                bond_tmp =[atomi, atomj, R[0],R[1],R[2]]
                bond_tmp_xc = [atomj, atomi, -R[0],-R[1],-R[2]]
                if not(bond_tmp_xc in bonds_rd) and not(bond_tmp in bonds_rd):
                    bonds_rd.append(bond_tmp)
        
            self.Bonds = np.asarray(bonds_rd)
        else:
            self.Bonds = np.asarray(bonds)
            
        self.NBonds = len(self.Bonds)
        
        # on site bond
        self.BondsOnSite = []
        for ii in range(self.NumAtoms):
            self.BondsOnSite.append([ii,ii,0,0,0])
        self.BondsOnSite = np.asarray(self.BondsOnSite)
        
    def PrintBondInfo(self):
        ilist = self.Bonds[:,0]
        # jlist = self.Bonds[:,1]
        print('# \t AtomID \t NBonds \t NOrbs')
        for ii in range(self.NumAtoms):
            print('# %12d%17d%15d' %(ii, len(ilist[ilist==ii]) + 1, 
                    self.AtomTypeNOrbs[self.TypeID[ii]]))



