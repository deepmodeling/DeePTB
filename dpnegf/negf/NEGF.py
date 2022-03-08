import time
from scipy.linalg.decomp import eig
import spglib
import numpy as np
from multiprocessing import Pool
import scipy.linalg as scl
import matplotlib.pyplot as plt
from negf.SurfaceGF import SurfGF

#from sktb.BuildHS import DeviceHS
#from sktb.GreenFunctions import SurfGF, SelfEnergy, NEGFTrans
#from sktb.ElectronBase import Electron

class NEGFcal(object):
    """calcuate Contact surface green's function, transport self energy, and then NEGF Transmission

    Input
    -----
    Processors: num of prcessors.
    save tags: SaveSurface, SaveSelfEnergy, SaveTrans

    eta: the imaginary part in both surface GF and self energy
    max_iteration: iteration number
    epsilon: convergence criterion.

    Attributes
    Scat_Hamiltons: function to input the hamiltonian of scatter region and interaction b/w scatter  and contact.
    Cont_Hamiltons: function to input the two principal layer hamiltonians:  H00 H01.
    Cal_Transmission: function to calculate the transmission.
    Cal_SurfGF: function to calculate surface GF.
    Cal_SelfEnergy: function to calculate self energy.
    """
    def __init__(self,paras):
        self.Processors = paras.Processors
        self.SaveSurface = paras.SaveSurface
        self.SaveSelfEnergy = paras.SaveSelfEnergy
        self.CalTrans = paras.CalTrans

        self.Contacts = paras.Contacts
        # self.ContactsNames = paras.ContactsNames
        self.ContactsNames = [item.lower() for item in self.Contacts]

        self.eta =  paras.eta
        self.max_iteration = paras.max_iteration
        self.epsilon = paras.epsilon
        
        self.Emin = paras.Emin
        self.Emax = paras.Emax
        self.NumE = paras.NumE

        self.CalDeviceDOS = paras.CalDeviceDOS
        self.CalDevicePDOS = paras.CalDevicePDOS


    def Scat_Hamiltons(self,HamilDict,Efermi=0):
        """  input the hamiltonian of scatter region.
        
        Input
        ------
        HamilDict: dictionary object contains the hamiltonian for each key.
        """
        # The shape of H should be [nk, norb1, norb2]
        self.Hss = HamilDict['Hss']
        self.scat_klist = HamilDict['klist']

        if 'Sss' in HamilDict.keys():
            self.Sss = HamilDict['Sss']
        else:
            self.Sss = np.zeros_like(self.Hss)
            self.Sss[:] = np.eye(self.Hss.shape[1])
        
        self.Hss -= Efermi * self.Sss

    def Scat_Cont_Hamiltons(self,HamilDict,Efermi=0):
        """  input the hamiltonian of interaction b/w scatter  and contact.
        
        Input
        ------
        HamilDict: dictionary object contains the hamiltonian for each key.
        """
        
        self.Hsc = {}
        self.Ssc = {}
        for itag in self.Contacts:
            self.Hsc[itag] =  HamilDict[itag]['Hsc']
            if 'Ssc' in HamilDict[itag].keys():
                self.Ssc[itag] =  HamilDict[itag]['Ssc']
            else:
                self.Ssc[itag] = np.zeros_like(self.Hsc[itag])
            
            if type(Efermi)==dict:
                self.Hsc[itag] -= Efermi[itag] * self.Ssc[itag]
            else:
                self.Hsc[itag] -= Efermi * self.Ssc[itag]

    def Cont_Hamiltons(self,HamilDict,Efermi=0):
        """  input the hamiltonian contact.
        
        Input
        ------
        HamilDict: dictionary object contains the hamiltonian for each key.
        """
        # The shape of H should be [nk, norb1, norb2]
        self.H00 = {}
        self.H01 = {}
        self.S00 = {}
        self.S01 = {}
        
        self.cont_klist = HamilDict['klist']
        # assert (contklist - self.klist < 1e-5).all()

        for itag in self.Contacts:
            self.H00[itag] =  HamilDict[itag]['H00']
            self.H01[itag] =  HamilDict[itag]['H01']

            if 'S00' in HamilDict[itag].keys():
                self.S00[itag] =  HamilDict[itag]['S00']
                self.S01[itag] =  HamilDict[itag]['S01']
            
            else:
                self.S00[itag] = np.zeros_like(self.H00[itag])
                self.S00[itag][:] = np.eye(self.H00[itag].shape[1])
                self.S01[itag] = np.zeros_like(self.H01[itag])
            
            if type(Efermi)==dict:
                self.H00[itag] -= Efermi[itag] * self.S00[itag]
                self.H01[itag] -= Efermi[itag] * self.S01[itag]
            else:
                self.H00[itag] -= Efermi * self.S00[itag]
                self.H01[itag] -= Efermi * self.S01[itag]

        # self.BZkpoints = SurfGF.MPSampling(size = paras.kmesh)


    def Cal_NEGF(self):

        self.ContGF = {}
        Elist = np.linspace(self.Emin,self.Emax,self.NumE)
        tst  = time.time()
        for itag in self.Contacts:
            bulkGF, topsurfGF, botsurfGF = self.Cal_SurfGF(Emin=self.Emin, Emax=self.Emax, tag=itag)
            ted =  time.time()
            print("# " + itag + " surface GF : %12.2f sec."  %(ted-tst))
            tst = ted
            self.ContGF[itag] = {'Bulk':bulkGF,'Surf':topsurfGF}
        
        if self.SaveSurface:
            savesurf = {}
            savesurf['Elist'] = Elist
            for itag in self.Contacts:
                BulkSpectralFunc = np.trace(self.ContGF[itag]['Bulk'], axis1=2, axis2=3)
                SurfSpectralFunc = np.trace(self.ContGF[itag]['Surf'], axis1=2, axis2=3)

                BulkSpectralFunc = -1*BulkSpectralFunc.imag / np.pi
                SurfSpectralFunc = -1*SurfSpectralFunc.imag / np.pi
                savesurf[itag] = {'Bulk':BulkSpectralFunc,'Surf':SurfSpectralFunc}
            np.save('SurfaceGF.npy',savesurf)
        print('# Finish surface green function calculations.')

        self.SelfEs = []
        for itag in self.Contacts:
            SelfEs = self.Cal_SelfEnergy(Emin=self.Emin, Emax=self.Emax, tag=itag)
            self.SelfEs.append(SelfEs)
        self.SelfEs =np.asarray(self.SelfEs)
        print ('# Finish Self Energy calculations.')


        self.GS = self.Cal_Gs(Emin=self.Emin, Emax=self.Emax)
        print('# Finish Green Functions Calculations.')

        # add save:
        np.save('GS',self.GS)
        np.save('SelfE',self.SelfEs)
        np.save('Hss',self.Hss)
        np.save('Sss',self.Sss)

        # Device density of states Tr{GS*S}
        if self.CalDeviceDOS:
            dos_device = []
            if self.CalDevicePDOS:
                pdos_device = []
            for ik in range(len(self.scat_klist)):
                dos_device_e = []
                if self.CalDevicePDOS:
                    pdos_device_e = []
                for ie in range(self.NumE):
                    GretardS = np.dot(self.GS[ik,ie],self.Sss[ik]) 
                    dos_tmp = -1 * np.trace(GretardS.imag)/np.pi
                    dos_device_e.append(dos_tmp)
                    
                    if self.CalDevicePDOS:
                        SGretardS = np.dot(self.Sss[ik],GretardS)
                        pdos_tmp = -(SGretardS.diagonal() / self.Sss[ik].diagonal()).imag / np.pi
                        pdos_device_e.append(pdos_tmp)
                dos_device.append(dos_device_e)
                if self.CalDevicePDOS:
                    pdos_device.append(pdos_device_e)

            self.dos = np.asarray(dos_device)
            np.save('DeviceDOS.npy',{'Energy':Elist,'DOS':self.dos})
            print('# Finish DeviceDOS.')
            if self.CalDevicePDOS:
                self.pdos = np.asarray(pdos_device)
                np.save('DevicePDOS.npy',{'Energy':Elist,'PDOS':self.pdos})
                print('# Finish DevicePDOS.')

        """
        if self.CalDevicePDOS:
            pdos_device= []
            for ik in range(len(self.scat_klist)):
                dos_device_e =[]
                for ie in range(self.NumE):
                    GretardS = np.dot(self.GS[ik,ie],self.Sss[ik])                    
                    SGretardS = np.dot(self.Sss[ik],GretardS)
                    dos_tmp = -(SGretardS.diagonal() / self.Sss[ik].diagonal()).imag / np.pi
                    dos_device_e.append(dos_tmp)
                pdos_device.append(dos_device_e)
            pdos_device = np.asarray(pdos_device)
            np.save('DevicePDOS.npy',{'Energy':Elist,'DOS':pdos_device})
            print('# Finish Transimission calculations.')
        """

        if self.CalTrans:
            self.trans = self.Cal_Trans()
            np.save('Transmission.npy',{'Energy':Elist,'Trans':self.trans})
            print('# Finish Transimission calculations.')


        #with Pool(processes=self.Processors) as pool:
        #    poolresults = pool.map(self.trans.GFScatterEne, self.surf.Elist)
        #self.GS = np.asarray(poolresults)
    
    
    def Cal_SurfGF(self,
                    Emin,
                    Emax,
                    EF=0.0,
                    tag='Source'): 
        """ calculate surface green's function. G(k,w) k is k-kpoint, w is energy.
        
        Input
        ----- 
        Emin: the minimum of energy
        Emax: the maximum of energy
        tag: the contact to be calculated

        Return 
        ------ 
        bulkGF: bulk
        topsurfGF: top surface 
        botsurfGF: bottom surface
        """
        assert(tag.lower() in self.ContactsNames)
        #tstat = time.time()
        Elist = np.linspace(Emin,Emax,self.NumE) - EF

        bulkGF=[]
        topsurfGF=[]
        botsurfGF=[]
        for ik in range(len(self.cont_klist)):
            self.H00ik = self.H00[tag][ik]
            self.H01ik = self.H01[tag][ik]
            self.S00ik = self.S00[tag][ik]
            self.S01ik = self.S01[tag][ik]
            with Pool(processes=self.Processors) as pool:
                poolresults = pool.map(self.Surface_GF_E, Elist)

            poolresults = np.asarray(poolresults)
            bulkGFik = poolresults[:,0]
            topsurfGFik = poolresults[:,1]
            botsurfGFik = poolresults[:,2]

            bulkGF.append(bulkGFik)
            topsurfGF.append(topsurfGFik)
            botsurfGF.append(botsurfGFik)

        bulkGF = np.array(bulkGF)
        topsurfGF = np.array(topsurfGF)
        botsurfGF = np.array(botsurfGF)

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

    def Surface_GF_E(self,E):
        Ene = E
        bulkgf, surfgf_top, surfgf_bot = \
                    SurfGF.iterationGF(self.H00ik, self.H01ik, self.S00ik,self.S01ik, Ene,\
                        self.eta, self.max_iteration, self.epsilon)
        
        return np.asarray([bulkgf, surfgf_top, surfgf_bot])

    
    def Cal_SelfEnergy(self,
                        Emin,
                        Emax,
                        EF=0.0,
                        tag='Source'):
        """ ccalculate  self energy. Sigma(k, w).
        
        Input
        -----
        Emin: the minimum of energy
        Emax: the maximum of energy
        tag: the contact to be calculated

        Return 
        ------ 
        SigmaKE: Sigma(k, w).
        """
        assert(tag.lower() in self.ContactsNames)
        if not ((self.scat_klist - self.cont_klist)<1.0E-6).all():
            raise ValueError('The k list in scatter and contact must be the same.')
        
        Elist = np.linspace(Emin,Emax,self.NumE) - EF
        
        SigmaKE = []
        for ik in range(len(self.scat_klist)):
            # self.Hssik = self.Hss[ik]
            self.Hscik = self.Hsc[tag][ik]
            # self.Sssik = self.Sss[ik]
            self.Sscik = self.Ssc[tag][ik]
            

            #self.SurfG00(self.ContGF[tag]['Surf'])
            # self.pot = self.ContactsPot[tag]

            #with Pool(processes=self.Processors) as pool:
            #    poolresults = pool.map(self.selfene.SelfEne, self.surf.Elist)
            #SigmaE = np.asarray(poolresults)
            
            SigmaE = []
            for ie in range(self.NumE):
                Ene = Elist[ie]
                self.G00ikE = self.ContGF[tag]['Surf'][ik,ie]
                res = self.SelfEnergy_E(Ene)
                SigmaE.append(res)
            SigmaKE.append(SigmaE)

        SigmaKE = np.asarray(SigmaKE)

        return SigmaKE

    def SelfEnergy_E(self,E):
        G00 = self.G00ikE
        E_eta = E + 1.0j*self.eta
        Vint  = E_eta * self.Sscik - self.Hscik
        Vint_ct = np.transpose(np.conjugate(Vint))
        Sigma_p =  np.dot(Vint,G00)
        Sigma_E =  np.dot(Sigma_p,Vint_ct)
        return Sigma_E


    def Cal_Gs(self,
                Emin,
                Emax,
                EF=0.0):
        Elist = np.linspace(Emin,Emax,self.NumE) - EF

        GsKE=[]
        for ik in range(len(self.scat_klist)):
            self.Hssik = self.Hss[ik]
            self.Sssik = self.Sss[ik]

            GsE=[]
            for ie in range(self.NumE):
                Ene = Elist[ie]
                self.SelfE_k_E =  self.SelfEs[:,ik,ie]
                gse = self.GS_E(Ene)
                GsE.append(gse)
            GsKE.append(GsE)

        GsKE = np.asarray(GsKE)
        return GsKE


    def GS_E (self,E):
        E_eta = E + 1.0j * self.eta
        # self.SelfE(:,ik,ie)
        gsH = E_eta * self.Sssik -self.Hssik - np.sum(self.SelfE_k_E[:],axis=0)
        Gs_E = np.linalg.inv(gsH)
        return Gs_E


    def Cal_Trans(self):
        trans = []
        trans = []
        for ik in range(len(self.scat_klist)):
            trans_E = []
            for ie in range(self.NumE):
                self.GS_ikE = self.GS[ik,ie,:]
                self.SelfEs_ikE = self.SelfEs[:,ik,ie]
                trans_ikE = self.Trans_E()
                trans_E.append(trans_ikE)
            trans.append(trans_E)
        trans = np.asarray(trans)

        return trans
        

    def Trans_E(self):
        gs = self.GS_ikE
        SelfEs = self.SelfEs_ikE
        num_cont = len(SelfEs)
        gsct = np.transpose(np.conjugate(gs))
        Gamma = []
        for ii in range(num_cont):
            broden = 1.0j * (SelfEs[ii] - np.transpose(np.conjugate(SelfEs[ii])))
            Gamma.append(broden)
        Gamma = np.asarray(Gamma)
        
        trans_E = []
        for ii in range(num_cont):
            gsG = np.dot(gs,Gamma[ii])
            gsGgs = np.dot(gsG,gsct)
            for jj in range(num_cont):
                gsGgsG = np.dot(gsGgs,Gamma[jj])
                
                trans_tmp = np.trace(gsGgsG)
                trans_E.append(trans_tmp)
        trans_E = np.asarray(trans_E)
        return trans_E