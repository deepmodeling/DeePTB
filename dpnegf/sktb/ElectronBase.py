import numpy  as np
import scipy.linalg as scl
import spglib
import matplotlib.pyplot as plt

from sktb.StructData import StructBuild,BondListBuild
from sktb.SlaterKosterPara import SlaterKosterInt
from sktb.BuildHS import BuildHSreal

class Electron(BuildHSreal):
    def __init__(self,paras):
        struct = StructBuild(paras)
        struct.IniUsingFile(StructFileName = paras.StructFileName,
                    StructFileFmt = paras.StructFileFmt)
        bondlst = BondListBuild(paras)
        bondlst.BondStuct(struct.Struct)
        bondlst.GetBonds()
        bondlst.PrintBondInfo()
        SKint = SlaterKosterInt(paras)
        SKint.ReadSKfiles()
        SKint.IntpSKfunc()
        BuildHSreal.__init__(self, bondlst)
        #self.HSreal = BuildHSreal(bondlst)
        self.HSMatBuild(SKint)
        # elec = Electron(bondlst)
        # elec.BuildHSkrecp(HSreal)
        #   self.Bonds     = BondBuildIns.Bonds
        #   self.NBonds    = BondBuildIns.NBonds
        #   self.StructAse = BondBuildIns.Struct
        #   self.Lattice   = BondBuildIns.Lattice
        #   self.TypeID    = BondBuildIns.TypeID
        #   self.AtomTypeNOrbs = BondBuildIns.AtomTypeNOrbs
        #   self.ProjAnglrM  = BondBuildIns.ProjAnglrM
        #   self.AnglrMID    = BondBuildIns.AnglrMID
        #   self.BondsOnSite = BondBuildIns.BondsOnSite
        #   self.NumAtoms    = BondBuildIns.NumAtoms  
        self.Struct   = struct.Struct
        self.RecpLatt = 2*np.pi * np.transpose(np.linalg.inv(self.Lattice))
        self.KPATH    = paras.KPATH
        self.NKpLine  = paras.NKpLine
        self.KPATHstr = paras.KPATHstr
        self.BZmesh   = paras.BZmesh
        self.ValElec  = np.array(bondlst.ProjValElec)
        self.DegSpin  = 2

    #def Ham_real2k(self, hij_all, bond, kpath, num_orbs):
    #def BuildHSkPath(self):

        #HKsite = self.Hamilreal2K(HSreal.BondHsite, HSreal.BondsOnSite, klists, HSreal.AtomNOrbs)
        #SKsite = self.Hamilreal2K(HSreal.BondSsite, HSreal.BondsOnSite, klists, HSreal.AtomNOrbs)
    
    def CalMeshEig(self):
        self.irkmesh, self.irkweigh = self.BZSampling(structase=self.Struct, mesh=self.BZmesh)
        HamilK = self.Hamilreal2K(self.BondHBlock, self.Bonds, self.irkmesh, self.AtomNOrbs)
        SoverK = self.Hamilreal2K(self.BondSBlock, self.Bonds, self.irkmesh, self.AtomNOrbs)
        eigs=[]
        for i in range(len(self.irkmesh)):
            eig_k,eig_vec= scl.eigh(HamilK[i],SoverK[i])
            eigs.append(eig_k * 13.605662285137 * 2)
        self.mesheig = np.asarray(eigs)
    
    
    def GetFermi(self):
        if not hasattr(self,'mesheig'):
            self.CalMeshEig()
        Numelecs = np.sum(self.ValElec[self.TypeID])
        NumirK = len(self.irkmesh)
        upband = NumirK * Numelecs//self.DegSpin
        dnband = NumirK * Numelecs//self.DegSpin - 1
        eigsort1d = np.sort(np.reshape(self.mesheig,[-1]))
        self.Efermi = (eigsort1d[dnband] + eigsort1d[upband])/2
        
        print('# Efermi : %16.6f' %self.Efermi)
        return self.Efermi
    @staticmethod
    def BZSampling(structase, mesh):
        symmetry = spglib.get_spacegroup(structase, symprec=1e-5)
        mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, structase, is_shift=[0, 0, 0])
        print('#'+'-'*60)
        print('#     The space goup of the is ' + symmetry)
        print("#             k-mesh : ", mesh)
        print('#     Mapping between full and ir kpoints')
        for i, (ir_gp_id, gp) in enumerate(zip(mapping, grid)):
            kps = gp.astype(float) / mesh
            print("#%5d   -->>%5d :  ( %8.5f%10.5f%10.5f )" \
                            % (i, ir_gp_id, kps[0],kps[1],kps[2]))
        print('#'+'-'*60)
        unqind,counts = np.unique(mapping,return_counts=True)
        irkmesh = grid[unqind]/mesh
        kweight = counts/len(mapping)
        print('#'+'-'*60)
        outstring="# Number of ir-kpoints: %d "
        print(outstring  % len(unqind))
        for ik in range(len(unqind)):
            print("# (%8.5f%10.5f%10.5f ) %15.9f" \
                %(irkmesh[ik][0],irkmesh[ik][1],irkmesh[ik][2],kweight[ik]))
        print('#'+'-'*60)
        return irkmesh, kweight

    def GetBand(self):
        print('#'+'-'*60)
        print('# Band structure calculations.')
        print('# K-PATH : ')
        for ik in range(len(self.KPATH)-1):
            fmtstring = '# '+ self.KPATHstr[ik] +': [%8.6f %9.6f %9.6f] -> '+self.KPATHstr[ik+1]+': [%8.6f %9.6f %9.6f]'
            print( fmtstring %(self.KPATH[ik][0],self.KPATH[ik][1],self.KPATH[ik][2],
            self.KPATH[ik+1][0],self.KPATH[ik+1][1],self.KPATH[ik+1][2]))
        print('#'+'-'*60)
        self.klists = self.IntpKpath(self.KPATH,self.NKpLine)
        self.xlist, self.high_sym_kpoints = self.Klist2Xlist(self.klists,self.NKpLine,self.RecpLatt)
        HamilK = self.Hamilreal2K(self.BondHBlock, self.Bonds, self.klists, self.AtomNOrbs)
        SoverK = self.Hamilreal2K(self.BondSBlock, self.Bonds, self.klists, self.AtomNOrbs)
        eig=[]
        for i in range(len(self.klists)):
            eig_k,eig_vec= scl.eigh(HamilK[i],SoverK[i])
            eig.append(eig_k * 13.605662285137 * 2)
        self.eig = np.asarray(eig)

    def BandPlot(self, FigName= 'band.png', Emax = 1000, Emin = -1000, Efermi = 0, GUI = True, restart = True):
        if Emax == 1000:
            Emax = np.max(self.eig)
        if Emin == -1000:
            Emin = np.min(self.eig)
        if restart:
            assert(hasattr(self,'eig'))
        else:
            # self.BuildHSkPath()
            self.GetBand()


        np.save("Band.npy",{'klist':self.klists,'xlist':self.xlist,\
                'highsymkps':self.high_sym_kpoints,'highsymstr':self.KPATHstr,'band':self.eig})

        Nbands = self.eig.shape[1]

        if not GUI:
            plt.switch_backend('agg')
        plt.figure(figsize=(6,6))
        for i in range(Nbands):
            plt.plot(self.xlist, self.eig[:,i]-Efermi, 'b')
        plt.axhline(0,c='r',ls='--')
        plt.tick_params(direction = 'in')
        plt.ylim(Emin,Emax)
        plt.xlim(self.xlist.min(), self.xlist.max())
        
        for i in self.high_sym_kpoints:
            plt.axvline(i,c='gray',ls='--')
        #plt.xticks(self.high_sym_kpoints, self.HighSymmKpName,fontsize=12)
        plt.xticks(self.high_sym_kpoints,self.KPATHstr,fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylabel('Energy (eV)',fontsize=12)
        plt.savefig(FigName,dpi=300)
        if GUI:
            plt.show()

    @staticmethod
    def Hamilreal2K(hij_all, bond, kpath, num_orbs):
        Hk=[]
        total_orbs = np.sum(num_orbs)
        for k in kpath:
            hk = np.zeros([total_orbs,total_orbs],dtype=complex)
            for ib in range(len(bond)):
                R = bond[ib,2:]
                i = bond[ib,0]
                j = bond[ib,1]

                ist = int(np.sum(num_orbs[0:i]))
                ied = int(np.sum(num_orbs[0:i+1]))
                jst = int(np.sum(num_orbs[0:j]))
                jed = int(np.sum(num_orbs[0:j+1]))
                if ib < len(num_orbs):
                    hk[ist:ied,jst:jed] += 0.5 * hij_all[ib] * np.exp(-1j * 2 * np.pi* np.dot(k,R))
                else:
                    hk[ist:ied,jst:jed] += hij_all[ib] * np.exp(-1j * 2 * np.pi* np.dot(k,R))
       
            hk = hk + np.transpose(np.conjugate(hk))
            #for i in range(total_orbs):
            #    hk[i,i] = hk[i,i] * 0.5
            Hk.append(hk)
        Hk = np.asarray(Hk)
        return Hk
    
    @staticmethod
    def IntpKpath(KPATH,NKpLine):
        N=len(KPATH)
        dim=len(KPATH[0])
        if type(NKpLine) == list:
            npoints_list = np.array(NKpLine,dtype=int)
        elif type(NKpLine) == int:
            npoints_list = np.array([NKpLine] * (N-1), dtype=int)    

        KPointLists=np.zeros([np.sum(npoints_list),dim])
        for i in range(N-1):
            for j in range(dim):
                temp = np.linspace(KPATH[i][j],KPATH[i+1][j],npoints_list[i])
                KPointLists[np.sum(npoints_list[0:i]) : np.sum(npoints_list[0:i+1]),j]=temp
        
        return KPointLists
    
    @staticmethod
    def Klist2Xlist(Klists, NKpLine, RecLatt):
        xlist = [0.0]
        for i in range(1, len(Klists)):
            # print(Klists[i].shape)
            # print(RecLatt.shape)
            # print(np.dot(Klists[i],RecLatt).shape)
            ka = np.dot(Klists[i],RecLatt).reshape(3)
            kb = np.dot(Klists[i-1],RecLatt).reshape(3)
            delta = np.sqrt(sum([(kb[j] - ka[j]) ** 2 for j in range(0, 3)]))
            if i % NKpLine ==0:
                delta=0
            temp = xlist[i - 1] + delta
            xlist.append(temp)
        xlist=np.asarray(xlist)
        high_sym_kpoints = [xlist[0]] + \
                           [xlist[i - 1] for i in range(NKpLine, len(xlist) + 1,
                                                        NKpLine)]
        return xlist, high_sym_kpoints

