import numpy as np
from negf.SurfaceGF import SurfGF
class Trans(object):
    """ To calculate the Transmission coefficient at 0-bias. 

    Inputs
    -----

    Return
    ------
    
    note: for TB there is no good way to calculate the Transmission coefficient at finite bias. 
    This is beacuse we donot have a strategy to self-confidently update our TB Hamiltonian. 
    """
    def __init__(self, 
                 Hss, 
                 Hsc, 
                 Sss, 
                 Ssc,
                 bondss, 
                 bondsc,
                 orbss, 
                 orbsc):
        self.Hss = Hss
        self.Hsc = Hsc
        self.Sss = Sss
        self.Ssc = Ssc
        self.bondss = bondss
        self.bondsc = bondsc
        self.orbss = orbss
        self.orbsc = orbsc
    
    def HamilSKrecp(self,kp):
        self.Hssk = SurfGF.Hamr2Hamk(hij = self.Hss,kpoint = kp, \
                    bond = self.bondss, orbsi = self.orbss, orbsj = self.orbss, conjtran = True)
        self.Sssk = SurfGF.Hamr2Hamk(hij = self.Sss,kpoint = kp, \
                    bond = self.bondss, orbsi = self.orbss, orbsj = self.orbss, conjtran = True)

        self.Hsck = SurfGF.Hamr2Hamk(hij = self.Hsc,kpoint = kp, \
                    bond = self.bondsc, orbsi = self.orbss, orbsj = self.orbsc, conjtran = False)
        self.Ssck = SurfGF.Hamr2Hamk(hij = self.Ssc,kpoint = kp, \
                    bond = self.bondsc, orbsi = self.orbss, orbsj = self.orbsc, conjtran = False)
    
    def Get_SelfEnergy(self, SigmaEs):
        self.SelfEs = SigmaEs
    
    def GetSelfEne(self, GS):
        self.GS = GS
        
    def GFScatterEne (self, ie):
        E_eta = self.Elist[ie] + 1.0j * self.Eta
        gsH = E_eta * self.Sssk -self.Hssk - np.sum(self.SelfEs[:,ie],axis=0)
        Gs_E = np.linalg.inv(gsH)
        return Gs_E
    
    def Transmission(self, ie):
        gs = self.GS[ie]
        SelfEs = self.SelfEs[:,ie]
        num_cont = len(SelfEs)
        gsct = np.transpose(np.conjugate(gs))
        Gamma = []
        for ii in range(num_cont):
            broden = 1.0j*(SelfEs[ii] - np.transpose(np.conjugate(SelfEs[ii])))
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


