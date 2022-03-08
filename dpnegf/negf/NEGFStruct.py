import numpy as np
import ase 
from sktb.StructData import StructBuild
from ase import Atoms

class StructNEGFBuild(StructBuild):
    """ Define the structure for negf calculations. 

    Attributes:
        Contacts: The contacts in the whole device.
        ContactsNames: format the contacts' name
        DeviceRegion: atom's indices in the central device region.
        ContactsRegions: atom's indices in contacts
        PrinLayNunit: unit cell given in the device stucture. can be 1  or 2.

    Functions:
        BuildRegions: build the stucture for each region in the device.
        BuildContact: build the contact region. input para: tag= cantact_name can be source or drain or any else defined in paras.Contacts
        BuildDevice: build the stucture for central devive.
    """

    def __init__(self, paras, StructAse):
        # print('# init StructNEGFBuild calss.')
        super(StructNEGFBuild,self).__init__(paras)
        #self.IniUsingFile(StructFileName = paras.StructFileName,StructFileFmt = paras.StructFileFmt)
        self.IniUsingAse(StructAse = StructAse)
        self.DeviceRegion = paras.DeviceRegion
        self.Contacts = paras.Contacts
        self.ContactsRegions = paras.ContactsRegions
        self.PrinLayNunit = paras.PrinLayNunit
        self.ContactsNames = [item.lower() for item in self.Contacts]
    
    def BuildRegions(self):
        """ build every regions to ase stucture object.
        
        Attributes
        -----------
        ExtDirect: dict, for each contact the transport direction index 0 - x, 1 - y, 2 - z.
        ContactStruct: dict, ASE Atoms structure  for each contact.
        ContactsExtSign: dict, iteration direction for each contact.
        GeoRegions:  the range of atomic index in different regions -- contact, device. 
        GeoStruct: the whole device ASE Atoms structure. central region with 2 PLs for each contact.
        """
        self.ExtDirect       = {} 
        self.ContactStruct   = {}
        self.ContactsExtSign = {}
        for itag in self.Contacts:
            ExtDirection, Rsign, ContactStruct = self.BuildContact(tag = itag)
            self.ContactsExtSign[itag] = Rsign
            self.ContactStruct[itag] = ContactStruct
            self.ExtDirect[itag] = ExtDirection
            # Device and scatter regions.
        self.GeoRegions, self.GeoStruct = self.BuildDevice(tag='All')
        
        # To obtain the projected structure and projection regions.
        # 1. initial structe class 
        self.IniUsingAse(StructAse = self.GeoStruct)
        # 2. call projection function to get the self.ProjStruct
        self.Projection()
        # 3. Using the self.ProjAtoms to obtain the projregions.
        self.ProjRegions={}
        ist = 0
        ied = 0
        for itag in ['Device'] + list(self.Contacts):
            ist = ied +1
            OriRegion = np.array(self.GeoRegions[itag]) - 1 
            projind = self.ProjAtoms[OriRegion[0]:OriRegion[1]+1]
            ied  = np.sum(projind) + ist - 1
            self.ProjRegions[itag] = [ist,ied]

        ase.io.write('OutDeviceStr.extxyz',self.GeoStruct,format='extxyz')
        print('#'+'-'*60)
        print('#' + ' print device struct in extxyz format')
        for ik in self.GeoRegions.keys():
            print("# \'"+ik+"\'" + ' atom ranges: ', self.GeoRegions[ik])
        for itag in self.Contacts:
            print('#' + ' print ' + itag + ' struct in extxyz format')
            ase.io.write(itag+'Str.extxyz',self.ContactStruct[itag],format='extxyz')
        print('#'+'-'*60)


    def BuildContact(self,tag='Source'):
        """ build contact ase structs for gaven tag source or drain or else defined in ContactsNames.

        Parameters
        ----------
        tag: elements in list ContactsNames. default source or drain.

        Return
        ------
        ExtendDirectInd:  the transport direction index 0 - x, 1 - y, 2 - z.
        Rsign: the direction of iteration in surface green's function calculations.
        ContactStruct: the contact ase struct with 1 principal layer.
        """

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
        symbols = self.Struct.symbols[Atomrange[0]:Atomrange[1]+1]
        unitstmbol = symbols[0 : NatomsContLayer]

        diffv = unitlayer2 - unitlayer1
        # diffv[0] :  pos[0] + R - pos[0] 
        assert(np.isclose(diffv[:],diffv[0],atol=1e-06).all())
        """
        To get the transport direction: ExtendDirect. the transport direction is the direction contact extends to  infinity.
        this only works for the rectangle lattice cell, which is most cases in NEGF calculations.
        """
        ExtendDirect = [not item for item in np.isclose(diffv[0],0,atol=1e-06)]
        assert(np.sum(ExtendDirect)==1)
        ExtendDirectInd = np.array([0,1,2])[ExtendDirect][0]

        PeriodicCond = [False, False, False]
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
        # Volume = np.dot(Lattice[0], np.cross(Lattice[1],Lattice[2]))
        # assert(Volume>0)
        """
        This is to get the direction for iteration to get surface green fucntion
        In most cases, the top and bottom surface green function is different 
        since they have different dangling bonds.
        """
        Rsign = int(np.sign(np.dot(diffv[0],Lattice[ExtendDirectInd])))
        # Contactstruct = Atoms(symbols=unitstmbol,positions=unitlayer1,
        # pbc=PeriodicCond,cell=Lattice)

        PLpositions = np.copy(unitlayer1)
        PLNunits = self.PrinLayNunit[contname]
        """
        Control how many unit cells are included as principal layer (PL).
        """
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
        """ build  "Scatter" or the "whole" device ase structs.

        Parameters
        ----------
        tag: Scatter or all

        Return
        ------
        RegionRanges: The range of atomic index in different regions -- contact, device. 
        DeviceStruct: The device ase stucture. 
            when tag=all, central  material + contact with 2 PLs
            when tag=scatter, central  material + contact with 1 PLs
        """ 

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
        RegionRanges = {'Device':self.DeviceRegion}
        for itag in self.Contacts:
            Rsign     = self.ContactsExtSign[itag]
            ExtDirect = self.ExtDirect[itag]
            contstr   = self.ContactStruct[itag]
            ist = ied +1
            ied = ist + len(contstr.positions)*NPL - 1     
            RegionRanges[itag] = [ist,ied]
            ScatterSymbols = ScatterSymbols + list(contstr.symbols) * NPL
            for ipl in range(NPL):
                ScatterPositions = np.concatenate([ScatterPositions, 
                    contstr.positions + ipl * Rsign * contstr.cell[ExtDirect]],axis=0)
            ScatterLattice[ExtDirect] = 0.0
            PeriodicCond[ExtDirect]   = False
        DeviceStruct = Atoms(symbols=ScatterSymbols, positions=ScatterPositions,
                                 cell=ScatterLattice, pbc=PeriodicCond)

        return RegionRanges, DeviceStruct



