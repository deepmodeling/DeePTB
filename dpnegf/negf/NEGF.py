import time
from scipy.linalg.decomp import eig
from scipy.constants import Boltzmann as kB
import numpy as np
from multiprocessing import Pool
from dpnegf.negf.SurfaceGF import SurfGF

#from sktb.BuildHS import DeviceHS
#from sktb.GreenFunctions import SurfGF, SelfEnergy, NEGFTrans
#from sktb.ElectronBase import Electron

# The class is used to calculate the surface green's function, self energy, and then the transmission
class NEGFcal(object):
    """calcuate Contact surface green's function, transport self energy, and then NEGF Transmission

    Args:
        object (object): _description_
    """    
    def __init__(self,paras):
        self.Processors = paras.Processors
        self.Contacts = paras.Contacts
        # self.ContactsNames = paras.ContactsNames
        self.ContactsNames = [item.lower() for item in self.Contacts]

        self.eta =  paras.eta
        self.max_iteration = paras.max_iteration
        self.epsilon = paras.epsilon
        
        self.Emin = paras.Emin
        self.Emax = paras.Emax
        self.NumE = paras.NumE
        self.energies =  np.linspace(self.Emin,self.Emax,self.NumE)
        self.updata_e = False
        self.bias = np.linspace(paras.BiasV[0],paras.BiasV[1],paras.NumV)

        self.CalDeviceDOS = paras.CalDeviceDOS
        self.CalDevicePDOS = paras.CalDevicePDOS


    def Scat_Hamiltons(self,HamilDict,Efermi=0):
        """input the hamiltonian of scatter region.

        The function `Scat_Hamiltons` takes a dictionary object `HamilDict` as input, and stores the
        Hamiltonian and overlap matrices in the `Hss` and `Sss` attributes of the `Scat_Region` object
        
        Args:       
            HamilDict (dict[str,np.array]): dictionary object contains the hamiltonian and overlap with shape [nk, norb, norb].
            Efermi (int, optional): Fermi energy, defaults to 0.
        """        
        self.Hss = HamilDict['Hss']
        self.scat_klist = HamilDict['klist']

        if 'Sss' in HamilDict.keys():
            self.Sss = HamilDict['Sss']
        else:
            self.Sss = np.zeros_like(self.Hss)
            self.Sss[:] = np.eye(self.Hss.shape[1])
        
        self.Hss -= Efermi * self.Sss

    def Scat_Cont_Hamiltons(self,HamilDict,Efermi=0):
        """input the hamiltonian of interaction b/w scatter  and contact.
        
        The function `Scat_Cont_Hamiltons` takes a dictionary object `HamilDict` as input, and stores the dictionary of contacts.
        "source","drain" as keys,etc. for each HamilDict[key], it stores the hamiltonian and overlap matrices in the `Hsc` and
        `Ssc` attributes of the `Scat_Region` and `Cont_Region` coupling.

        Args:
            HamilDict (dict[str, dict[str,np.array]]): dictionary object contains the dictionary of 
                    the hamiltonian and overlap with shape [nk, norb1, norb2] for each key.
            Efermi (int, optional): Fermi energy, defaults to 0.
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
        """input the hamiltonian contact.
        
        The function `Cont_Hamiltons` takes in a dictionary of Hamiltonians and overlaps for each contact, "source","drain" as keys,etc.
        for each HamilDict[key], it and stores them in the `H00`, `H01`, `S00`, and `S01` attributes, which are the hamiltonian and overlap 
        of the principal layer H00 (S00), and coupling between two layers H01 (S01).
        
        Args:
            HamilDict (dict[str,dict[str,np.array]]): dictionary object contains the dictionary of hamiltonian and overlap 
                    for each key. the hamiltonian and overlap with shape [nk, norb1, norb1]. 
            Efermi (int, optional): Fermi energy, defaults to 0 (optional)
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

    def get_cont_greens(self,E=None,tag='Source'):
        """
        The function `get_cont_greens` calculates the contacts' surface Green's function for a given energy and contact
        
        :param E: the energy list
        :param tag: the name of the contact, defaults to source (optional)
        """

        assert(tag.lower() in self.ContactsNames)
        contind = self.ContactsNames.index(tag.lower())
        tag = self.Contacts[contind]

        if E is None:
            E = self.energies.copy()
        else:
            E = np.asarray(E)
            self.energies = E
            self.NumE = len(E)
                    
        # checking if the attributes bulkGF, topsurfGF, and botsurfGF exist. If they do not
        # exist, then they are created.
        if not hasattr(self,'bulkGF'):
            self.bulkGF = {}
        if not hasattr(self,'topsurfGF'):
            self.topsurfGF = {}
        if not hasattr(self,'botsurfGF'):
            self.botsurfGF = {}

        
        bulkGF=[]
        topsurfGF=[]
        botsurfGF=[]
        for ik in range(len(self.cont_klist)):
            self.ik = ik
            self.cal_tag = tag
            with Pool(processes=self.Processors) as pool:
                poolresults = pool.map(self._get_surf_greens_e, E)

            poolresults = np.asarray(poolresults)
            bulkGFik = poolresults[:,0]
            topsurfGFik = poolresults[:,1]
            botsurfGFik = poolresults[:,2]

            bulkGF.append(bulkGFik)
            topsurfGF.append(topsurfGFik)
            botsurfGF.append(botsurfGFik)

        self.bulkGF[tag] = np.array(bulkGF)
        self.topsurfGF[tag] = np.array(topsurfGF)
        self.botsurfGF[tag] = np.array(botsurfGF)
        self.updata_e = False

    def _get_surf_greens_e(self,energy):
        """This function calculates the surface Green's function for a given energy

        Args:
            energy (float): energy

        Returns:
            np.array([bulkgf, surfgf_top, surfgf_bot]) : The Green's function for the bulk, top surface, and bottom surface.
        """ 
        # define the self.ik and self.cal_tag first, before call this function.
        H00ik = self.H00[self.cal_tag][self.ik]
        H01ik = self.H01[self.cal_tag][self.ik]
        S00ik = self.S00[self.cal_tag][self.ik]
        S01ik = self.S01[self.cal_tag][self.ik]

        bulkgf, surfgf_top, surfgf_bot = \
                    SurfGF.iterationGF(H00ik, H01ik, S00ik, S01ik, energy,\
                        self.eta, self.max_iteration, self.epsilon)
        
        return np.asarray([bulkgf, surfgf_top, surfgf_bot])


    def get_selfenergies(self, energies=None, tag='Source'):
        """This function calculates the self energy of the system

        Args:
            E (list, optional): the energy at which to calculate the self-energy. Defaults to None.
            tag (str, optional): the contact to be calculated, defaults to source.

        Returns:
            self.SigmaKE (np.array) : Sigma(k, w). 
        """        
        assert(tag.lower() in self.ContactsNames)
        contind = self.ContactsNames.index(tag.lower())
        tag = self.Contacts[contind]

        if energies is None:
            energies = self.energies.copy()
        else:
            energies = np.asarray(energies)
            self.energies = energies
            self.NumE = len(energies)
            self.updata_e = True

        if not ((self.scat_klist - self.cont_klist)<1.0E-6).all():
            raise ValueError('The k list in scatter and contact must be the same.')

        # if (not self.updata_e) 
        # if (not self.updata_e) and hasattr(self,'selfE') and tag in self.selfE.keys():
        #    print('The surface green function is already calculated.')
        #    return 0

        if not hasattr(self,'selfenergy'):
            self.selfenergy = {}

        SigmaKE = []
        for ik in range(len(self.scat_klist)):
            self.ik = ik
            self.cal_tag = tag
            #self.SurfG00(self.ContGF[tag]['Surf'])
            # self.pot = self.ContactsPot[tag]

            with Pool(processes=self.Processors) as pool:
                poolresults = pool.map(self._get_selfenergy_e, energies)
            SigmaKE.append(np.asarray(poolresults))
        
        self.selfenergy[tag] = np.asarray(SigmaKE)
        
    
    def _get_selfenergy_e(self,energy):
        Hscik = self.Hsc[self.cal_tag][self.ik]
        Sscik = self.Ssc[self.cal_tag][self.ik]
        if (not self.updata_e) and hasattr(self,'topsurfGF') and self.cal_tag in self.topsurfGF.keys():
            assert np.round(energy,6) in np.around(self.energies,6), "for not updata the energies, the E must in the self.energies"
            ind = np.where(np.around(self.energies,6) == np.round(energy,6))[0][0]
            G00 = self.topsurfGF[self.cal_tag][self.ik,ind]
        else:
            [bulkgf, G00, surfgf_bot] = self._get_surf_greens_e(energy)
        
        E_eta = energy + 1.0j*self.eta
        Vint  = E_eta * Sscik - Hscik
        Vint_ct = np.transpose(np.conjugate(Vint))
        Sigma_p =  np.dot(Vint,G00)
        Sigma_E =  np.dot(Sigma_p,Vint_ct)
        return Sigma_E

    def get_scatt_greens(self, energies=None):
        if energies is None:
            energies = self.energies.copy()
        else:
            energies = np.asarray(energies)
            self.energies = energies
            self.NumE = len(energies)
            self.updata_e = True
        
        if  not hasattr(self,'selfenergy'):
            self.selfenergy={}

        scatt_gf = []
        sigmaek ={}
        for itag in self.Contacts:
            sigmaek[itag]=[]
        for ik in range(len(self.scat_klist)):
            #self.Hssik = self.Hss[ik]
            #self.Sssik = self.Sss[ik]
            self.ik = ik

            with Pool(processes=self.Processors) as pool:
                poolresults = pool.map(self._get_scatt_greens_e, energies)

            scatt_gf.append(np.asarray([arr[0] for arr in  poolresults]))
            for itag in self.Contacts:
                sigmaek[itag].append(np.asarray([arr[1][itag] for arr in  poolresults]))

        self.scatt_gf = np.asarray(scatt_gf)
        for itag in self.Contacts:
            self.selfenergy[itag] = np.asarray(sigmaek[itag])


    def _get_scatt_greens_e (self,energy):
        Hssik = self.Hss[self.ik]
        Sssik = self.Sss[self.ik]
        E_eta = energy + 1.0j * self.eta
        self_energy_sum = 0.0 
        self_energy_tag = {}
        for tag in self.Contacts:
            if (not self.updata_e) and hasattr(self,'selfenergy') and tag in self.selfenergy.keys():
                assert np.round(energy,6) in np.around(self.energies,6), "for not updata the energies, the E must in the self.energies"
                ind = np.where(np.around(self.energies,6) == np.round(energy,6))[0][0]
                self_energy = self.selfenergy[tag][self.ik,ind]
                self_energy_tag[tag] = self_energy
            else:
                self.cal_tag = tag
                self_energy = self._get_selfenergy_e(energy)
                self_energy_tag[tag] = self_energy
            self_energy_sum += self_energy

        gsH = E_eta * Sssik - Hssik - self_energy_sum
        scatt_gf_e = np.linalg.inv(gsH)
        return [scatt_gf_e, self_energy_tag]
    
    def get_transmission(self, energies=None, tag1='Source', tag2='Drain'):
        """This function calculates the transmission of the system

        Args:
            energies (list, optional): the energy at which to calculate the transmission. Defaults to None.
        """
        assert(tag1.lower() in self.ContactsNames)
        assert(tag2.lower() in self.ContactsNames)
        contind = self.ContactsNames.index(tag1.lower())
        tag1 = self.Contacts[contind]
        contind = self.ContactsNames.index(tag2.lower())
        tag2 = self.Contacts[contind]

        if energies is None:
            energies = self.energies.copy()
        
        else:
            energies = np.asarray(energies)
            self.energies = energies
            self.NumE = len(energies)
            self.updata_e = True
        
        self.transmission  = {}
        transmission = []
        for ik in range(len(self.scat_klist)):
            self.ik = ik
            self.tag1 = tag1
            self.tag2 = tag2
            with Pool(processes=self.Processors) as pool:
                poolresults = pool.map(self._get_transmission_e, energies)
            transmission.append(poolresults)
        # index 0  means the k index 0.
        # For negf device, most cases k mesh being the gamma point is enough.
        self.transmission[tag1 + 'to' + tag2] = np.asarray(transmission)[0]


    def _get_transmission_e(self,energy):
        if (not self.updata_e) and hasattr(self,'scatt_gf'):
            assert np.round(energy,6) in np.around(self.energies,6), "for not updata the energies, the E must in the self.energies"
            ind = np.where(np.around(self.energies,6) == np.round(energy,6))[0][0]
            scatt_gf_e = self.scatt_gf[self.ik,ind]
            selfenergy_e = {}
            for tag in self.Contacts:
                assert hasattr(self,'selfenergy') and tag in self.selfenergy.keys(), 'if hasattr scatt_gf, mush also hasattr selfenergy '
                selfenergy_e[tag] = self.selfenergy[tag][self.ik,ind]

        else:
            [scatt_gf_e, selfenergy_e] = self._get_scatt_greens_e(energy)
        
        num_cont = len(self.Contacts)

        trans_e = []
        #for ii in range(num_cont):
        #tag = self.Contacts[ii]
        broaden = 1.0j * (selfenergy_e[self.tag1] - selfenergy_e[self.tag1].T.conj())
        gsG = np.dot(scatt_gf_e,broaden)
        gsGgs = np.dot(gsG, scatt_gf_e.T.conj())
        #for jj in range(num_cont):
        #tag2 = self.Contacts[jj]
        broaden = 1.0j * (selfenergy_e[self.tag2] - selfenergy_e[self.tag2].T.conj())
        gsGgsG = np.dot(gsGgs,broaden)
        #trans_e.append(np.trace(gsGgsG))
        trans_e = np.trace(gsGgsG)

        return trans_e


    def get_current(self, bias=None,  T=0., energies=None, transmission=None, tag1 = 'Source', tag2='Drain'):
        assert(tag1.lower() in self.ContactsNames)
        assert(tag2.lower() in self.ContactsNames)
        # Getting the transmission of the two tags.
        contind = self.ContactsNames.index(tag1.lower())
        tag1 = self.Contacts[contind]
        contind = self.ContactsNames.index(tag2.lower())
        tag2 = self.Contacts[contind]
        tag = tag1 + 'to' + tag2

        if bias is None:
            bias = self.bias.copy()
        else:
            bias = np.asarray(bias)
            self.bias = bias

        if energies is None:
            energies = self.energies.copy()
            if transmission is None:
                if hasattr(self,'transmission') and tag in self.transmission.keys():
                    transmission = self.transmission[tag].copy()
                else:
                    self.get_transmission(tag1=tag1, tag2=tag2)
                    transmission = self.transmission[tag].copy()                
                
        else:
            energies = np.asarray(energies)
            self.energies = energies.copy()
            self.NumE = len(energies)
            self.updata_e = True
            self.get_transmission(energies, tag1, tag2)
            transmission = self.transmission[tag].copy()

        
        bias = bias[np.newaxis]
        energies = energies[:, np.newaxis]
        transmission = transmission[:, np.newaxis]
        f1 = self.fermidist(energies - bias/2., kB * T)
        f2 = self.fermidist(energies + bias/2., kB * T)
        self.current = np.trapz((f1 - f2) * transmission, x=energies, axis=0)


    def fermidist(self, energies, kT):
        """
        It returns the Fermi-Dirac distribution for a given set of energies and temperature.
        
        :param energies: a list of energies
        :param kT: the temperature in units of energy. (kT = kB * T, kB is Boltzmann constant)
        :return: The Fermi-Dirac distribution.
        """
        # the Fermi-Dirac distribution.
        assert kT >= 0., "Temperature must be positive!"
        if kT==0:
            return np.asfarray(energies <=0)
        else:
            return 1. / (1. + np.exp(energies / kT))
