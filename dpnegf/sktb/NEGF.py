import time
from scipy.linalg.decomp import eig
import spglib
import numpy as np
from multiprocessing import Pool
import scipy.linalg as scl
import matplotlib.pyplot as plt

from dpnegf.sktb.BuildHS import DeviceHS
from dpnegf.sktb.GreenFunctions import SurfGF, SelfEnergy, NEGFTrans
from dpnegf.sktb.ElectronBase import Electron


class NEGFCal(DeviceHS,NEGFTrans):
    def __init__(self,paras) -> None:
        #super(NEGFCal,self).__init__()
        DeviceHS.__init__(self, paras)
        NEGFTrans.__init__(self, paras)
        self.Processors = paras.Processors
        self.SaveSurface = paras.SaveSurface
        self.SaveSelfEnergy = paras.SaveSelfEnergy
        self.SaveTrans   = paras.SaveTrans
        self.BZ_sampling()
        self.ShowContactBand = paras.ShowContactBand
        if self.ShowContactBand in  self.Contacts:
            self.KPATH = paras.KPATH
            self.NKpLine = paras.NKpLine
            Lattice = np.array(self.ContactStruct[self.ShowContactBand].cell)
            self.RecpLatt = 2*np.pi * np.transpose(np.linalg.inv(Lattice))

        self.GUI = paras.GUI
        self.KPATHstr = paras.KPATHstr
        self.CalDeviceDOS = paras.CalDeviceDOS
        self.ShowContactDOS = paras.ShowContactDOS
        self.DOSKMesh = paras.DOSKMesh
        #self.ValElec  = np.array(paras.ValElec)
        self.ValElec  = np.array(self.ProjValElec)
        self.DegSpin=2
        if self.ShowContactDOS:
            self.EmaxDOS = paras.EmaxDOS
            self.EminDOS = paras.EminDOS
            self.NEDOS   = paras.NEDOS  
            self.ContDOSElist = np.linspace(self.EminDOS,self.EmaxDOS,self.NEDOS)
            self.Sigma = paras.Sigma

        #self.ContactsPot   = paras.ContactsPot

    def CalTrans(self):
        self.ContGF = {}
        tst  = time.time()
        for itag in self.Contacts:
            bulkGF, topsurfGF, botsurfGF = self.CalSurfGF(itag)
            ted =  time.time()
            print("# " + itag + " surface GF : %12.2f sec."  %(ted-tst))
            tst = ted
            self.ContGF[itag] = {'Bulk':bulkGF,'Surf':topsurfGF}
        
        if self.SaveSurface:
            savesurf = {}
            savesurf['Elist'] = self.Elist
            for itag in self.Contacts:
                BulkSpectralFunc = np.trace(self.ContGF[itag]['Bulk'], axis1=1,axis2=2)
                SurfSpectralFunc = np.trace(self.ContGF[itag]['Surf'], axis1=1,axis2=2)

                BulkSpectralFunc = -1*BulkSpectralFunc.imag / np.pi
                SurfSpectralFunc = -1*SurfSpectralFunc.imag / np.pi
                savesurf[itag] = {'Bulk':BulkSpectralFunc,'Surf':SurfSpectralFunc}
            np.save('SurfaceGF.npy',savesurf)
        print('# Finish surface green function calculations.')

        self.SelfEs = []
        for itag in self.Contacts:
            SelfEs = self.CalSelfE(tag=itag)
            self.SelfEs.append(SelfEs)
        self.SelfEs =np.asarray(self.SelfEs)
        print ('# Finish Self Energy calculations.')

        Hss,  Sss,  bondss,  AtomNorbsSS  = self.GenDeviceHamilReal()
        self.HamilSReal(Hss=Hss,Sss=Sss,bondss=bondss,orbss=AtomNorbsSS)
        self.HamilSKrecp(self.kpoints[0])

        GS = []
        for ii in range(self.NumE):
            GSE = self.GFScatterEne(ii)
            GS.append(GSE)
        self.GS = np.asarray(GS)
        print('# Finish Green Functions Calculations.')

        # Device density of states Tr{GS*S}
        if self.CalDeviceDOS:
            dos_device= []
            for ie in range(self.NumE):
                dos_tmp = -1 * np.trace(np.dot(self.GS[ie],self.Sssk))
                dos_device.append(dos_tmp)
            dos_device = np.asarray(dos_device)
            np.save('DeviceDOS.npy',{'Energy':self.Elist,'DOS':dos_device})

        trans = []
        for ie in range(self.NumE):
            trans_E = self.Transmission(ie)
            trans.append(trans_E)
        self.trans = np.asarray(trans)
        if self.SaveTrans:
            np.save('Transmission.npy',{'Energy':self.Elist,'Trans':self.trans})

        print('# Finish Transimission calculations.')


        #with Pool(processes=self.Processors) as pool:
        #    poolresults = pool.map(self.trans.GFScatterEne, self.surf.Elist)
        #self.GS = np.asarray(poolresults)
    
    def CalFermiDos(self):
        for itag in self.Contacts:
            self.CalContDOS(tag=itag)
            
    def CalContDOS(self,tag='Source'):
        assert(tag.lower() in self.ContactsNames)
        # spglib.get_spacegroup(self.ContactStruct['Source'], symprec=1e-5)
        mesh = self.DOSKMesh[tag]
        crycell = self.ContactStruct[tag]
        symmetry = spglib.get_spacegroup(crycell, symprec=1e-5)
        mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, crycell, is_shift=[0, 0, 0])
        # All k-points and mapping to ir-grid points
        print('#'+'-'*60)
        print('#     The space goup of the ' + tag + ' is ' + symmetry)
        print('#     Mapping between full and ir kpoints for ' + tag)
        for i, (ir_gp_id, gp) in enumerate(zip(mapping, grid)):
            kps = gp.astype(float) / mesh
            print("#%5d   -->>%5d :  ( %8.5f%10.5f%10.5f )" \
                            % (i, ir_gp_id, kps[0],kps[1],kps[2]))
        print('#'+'-'*60)
        unqind,counts = np.unique(mapping,return_counts=True)
        irkmesh = grid[unqind]/mesh
        kweight = counts/len(mapping)
        print('#'+'-'*60)
        outstring="# Number of ir-kpoints: %d for " + tag
        print(outstring  % len(unqind))
        for ik in range(len(unqind)):
            print("# (%8.5f%10.5f%10.5f ) %15.9f" \
                %(irkmesh[ik][0],irkmesh[ik][1],irkmesh[ik][2],kweight[ik]))
        print('#'+'-'*60)

        H11, H12, S11, S12, bond11, bond12, AtomNorbs = \
                                self.GenContHamilReal(tag=tag)
        doseig=[]
        for ik in irkmesh:
            H11k = self.Hamr2Hamk(H11,ik,bond11,AtomNorbs,AtomNorbs,conjtran=True)
            S11k = self.Hamr2Hamk(S11,ik,bond11,AtomNorbs,AtomNorbs,conjtran=True)
            H12k = self.Hamr2Hamk(H12,ik,bond12,AtomNorbs,AtomNorbs,conjtran=True) 
            S12k = self.Hamr2Hamk(S12,ik,bond12,AtomNorbs,AtomNorbs,conjtran=True) 
            HamilK = H11k + H12k
            OverlapK = S11k + S12k
            eig_k,eig_vec= scl.eigh(HamilK,OverlapK)
            doseig.append(eig_k)
        self.doseig = np.asarray(doseig)
        typeid = []
        for i in self.ContactStruct[tag].symbols:
            typeid.append(self.Type2ID[i])
        Numelecs = np.sum(self.ValElec[typeid])
        
        NumirK = len(irkmesh)
        upband = NumirK * Numelecs//self.DegSpin
        dnband = NumirK * Numelecs//self.DegSpin - 1
        eigsort1d = np.sort(np.reshape(self.doseig,[-1]))
        self.Efermi = (eigsort1d[dnband] + eigsort1d[upband])/2
        print('# '+ tag + ' Efermi : %16.6f' %self.Efermi)

        if self.ShowContactDOS:
            eig =  self.doseig - self.Efermi
            upband = np.argmin(np.abs(np.min(eig,axis=0) - self.EmaxDOS - 1))
            dnband = np.argmin(np.abs(np.max(eig,axis=0) - self.EminDOS + 1))
            eigwindow = eig[:,dnband:upband]
            npickband = eigwindow.shape[1]
            dos = np.zeros(self.NEDOS)
            kweight = np.reshape(kweight,[-1,1])
            for ii in range(npickband):
                x = -1*(eigwindow[:,[ii]]  - self.ContDOSElist)**2/(2*self.Sigma**2)
                tmp = np.exp(x) * kweight/(self.Sigma*np.sqrt(2*np.pi))
                dos = dos + np.sum(tmp,axis=0)
            self.ContDOS = np.array(dos)

        # using gaussian to get dos.
    
    def CalContBand(self,tag ='Source'):
        assert(tag.lower() in self.ContactsNames)
        H11, H12, S11, S12, bond11, bond12, AtomNorbs = \
                                self.GenContHamilReal(tag=tag)
        #self.HamilContReal(H00r=H11, H01r=H12, S00r=S11, S01r=S12, \
        #    bond00 = bond11, bond01=bond12, num_orbs=AtomNorbs)
         
        self.klists = Electron.IntpKpath(self.KPATH,self.NKpLine)
        self.xlist, self.high_sym_kpoints = Electron.Klist2Xlist(self.klists,self.NKpLine,self.RecpLatt)
        eig=[]
        for ik in self.klists:
            H11k = self.Hamr2Hamk(H11,ik,bond11,AtomNorbs,AtomNorbs,conjtran=True)
            S11k = self.Hamr2Hamk(S11,ik,bond11,AtomNorbs,AtomNorbs,conjtran=True)
            H12k = self.Hamr2Hamk(H12,ik,bond12,AtomNorbs,AtomNorbs,conjtran=True) 
            S12k = self.Hamr2Hamk(S12,ik,bond12,AtomNorbs,AtomNorbs,conjtran=True) 
            HamilK = H11k + H12k
            OverlapK = S11k + S12k
            eig_k,eig_vec= scl.eigh(HamilK,OverlapK)
            eig.append(eig_k)
        self.eig = np.asarray(eig)
        np.save("Band.npy",{'klist':self.klists,'xlist':self.xlist,\
                'highsymkps':self.high_sym_kpoints,'highsymstr':self.KPATHstr,'band':self.eig})

        Nbands = self.eig.shape[1]
        if not self.GUI:
            plt.switch_backend('agg')
        plt.figure(figsize=(6,6))
        Efermi = -4.6822302838480425
        for i in range(Nbands):
            plt.plot(self.xlist, self.eig[:,i]-Efermi, 'b')

        plt.tick_params(direction = 'in')
        plt.ylim(-5,5)
        plt.xlim(self.xlist.min(), self.xlist.max())
        
        for i in self.high_sym_kpoints:
            plt.axvline(i,c='gray',ls='--')
        #plt.xticks(self.high_sym_kpoints, self.HighSymmKpName,fontsize=12)
        plt.xticks(self.high_sym_kpoints,self.KPATHstr,fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylabel('Energy (eV)',fontsize=12)
        plt.savefig('Contband.png',dpi=300)
        print("# Finish contact band calculations.")

        
    def CalSurfGF(self,tag='Source'): 
        assert(tag.lower() in self.ContactsNames)

        H11, H12, S11, S12, bond11, bond12, AtomNorbs = \
                                        self.GenContHamilReal(tag=tag)

        self.HamilContReal(H00r = H11, H01r = H12, S00r = S11, S01r = S12, \
            bond00 = bond11, bond01 = bond12, num_orbs = AtomNorbs)
        self.HamilContKrecp(kp = self.kpoints[0])
             #tstat = time.time()
        with Pool(processes=self.Processors) as pool:
            poolresults = pool.map(self.SurfGFE, self.Elist)
        poolresults = np.asarray(poolresults)
        bulkGF = poolresults[:,0]
        topsurfGF = poolresults[:,1]
        botsurfGF = poolresults[:,2]

        # bulkGF = []
        # topsurfGF = []
        # botsurfGF = []
        # for ie in range(self.NumE):
        #     Ene = self.Elist[ie]
        #     res = self.SurfGFE(Ene)
        #     bulkGF.append(res[0])
        #     topsurfGF.append(res[1])
        #     botsurfGF.append(res[2])
        # bulkGF = np.asarray(bulkGF)
        # topsurfGF = np.asarray(topsurfGF)
        # botsurfGF = np.asarray(botsurfGF)
        # # poolresults = np.asarray(poolresults)
        # #bulkGF = poolresults[:,0]
        # topsurfGF = poolresults[:,1]
        # botsurfGF = poolresults[:,2]
        return bulkGF, topsurfGF, botsurfGF

    def CalSelfE(self,tag='Source'):
        assert(tag.lower() in self.ContactsNames)
        Hsc, Ssc, bondsc, AtomNorbsS, AtomNorbsC = \
                            self.GenScatterHamilReal(tag=tag)

        self.HamilSCReal(Hsc=Hsc,Ssc=Ssc,bondsc=bondsc,\
                                        orbss=AtomNorbsS,orbsc=AtomNorbsC)
        self.HamilSCKrecp(self.kpoints[0])

        #self.SurfG00(self.ContGF[tag]['Surf'])
        self.G00 = self.ContGF[tag]['Surf']
        # self.pot = self.ContactsPot[tag]
        
        #with Pool(processes=self.Processors) as pool:
        #    poolresults = pool.map(self.selfene.SelfEne, self.surf.Elist)
        #SigmaE = np.asarray(poolresults)
        SigmaE = []
        for ie in range(self.NumE):
            #Ene = self.Elist[ie]
            res = self.SelfEne(ie)
            SigmaE.append(res)
        SigmaE = np.asarray(SigmaE)

        return SigmaE


