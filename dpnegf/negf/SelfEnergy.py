import numpy as np
from dpnegf.negf.SurfaceGF import SurfGF

class SelfEnergy(object):
    def __init__(self,Hsc, Ssc, bondsc, orbss, orbsc):
        self.Hsc    = Hsc
        self.Ssc    = Ssc
        self.bondsc = bondsc
        self.orbss  = orbss
        self.orbsc  = orbsc
        
    def HamilSCKrecp(self,kp):
        self.Hsck = SurfGF.Hamr2Hamk(hij = self.Hsc,kpoint = kp, \
                    bond = self.bondsc, orbsi = self.orbss, orbsj = self.orbsc,\
                                    conjtran = False)
        self.Ssck = SurfGF.Hamr2Hamk(hij = self.Ssc,kpoint = kp, \
                    bond = self.bondsc, orbsi = self.orbss, orbsj = self.orbsc,\
                                    conjtran = False)

    def SurfG00(self,G00):
        self.G00 = G00

    def SelfEne(self,ie):
        E = self.Elist[ie]
        G00 = self.G00[ie]
        E_eta = E + 1.0j*self.Eta
        Vint  = E_eta * self.Ssck - self.Hsck
        Vint_ct = np.transpose(np.conjugate(Vint))
        Sigma_p =  np.dot(Vint,G00)
        Sigma_E =  np.dot(Sigma_p,Vint_ct)
        return Sigma_E
