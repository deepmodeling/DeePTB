import numpy as np

class SurfGF(object):
    def __init__(self,paras):
        print('# init SurfGF calss.')
        self.max_iteration = 100
        self.epsilon = 1.0E-6
        #self.periodic = NEGFStruct.pbc
        self.Emin = paras.Emin
        self.Emax = paras.Emax
        self.NumE = paras.NumE
        self.Eta  = paras.Eta
        self.Elist = np.linspace(self.Emin,self.Emax,self.NumE)

    def BZ_sampling(self):
        #if self.periodic.all():
        #    print('Wrong setting for periodic! \n'+
        #          'Surface GF is for semi-infinite slab which must be non-periodic in at least direction')
        '''
        self.kpoints = []
        for ix in range(self.nkpoints[0]):
            for iy in range(self.nkpoints[1]):
                for iz in range(self.nkpoints[2]):
                    kptmp = np.zeros(3)
                    kptmp[0] = self.kpstart[0] + ix*self.kpvec1[0]/(self.nkpoints[0]) \
                                + iy*self.kpvec2[0]/(self.nkpoints[1]) + iz*self.kpvec3[0]/(self.nkpoints[2]);
                    kptmp[1] = self.kpstart[1] + ix*self.kpvec1[1]/(self.nkpoints[0]) \
                                + iy*self.kpvec2[1]/(self.nkpoints[1]) + iz*self.kpvec3[1]/(self.nkpoints[2]);
                    kptmp[2] = self.kpstart[2] + ix*self.kpvec1[2]/(self.nkpoints[0]) \
                                + iy*self.kpvec2[2]/(self.nkpoints[1]) + iz*self.kpvec3[2]/(self.nkpoints[2]);
                    self.kpoints.append(kptmp)
        '''
        # Gamma only! 
        self.kpoints = np.array([[0.0,0.0,0.0]])


        
    def Hamr2Hamk(self, hij, kpoint, bond, orbsi, orbsj, conjtran = False):
        k = kpoint
        norbsi  = np.sum(orbsi) 
        norbsj  = np.sum(orbsj) 
        hk = np.zeros([norbsi,norbsj],dtype=complex)
        for ib in range(len(bond)):
            R = bond[ib,2:]
            i = bond[ib,0]
            j = bond[ib,1]
            ist = int(np.sum(orbsi[0:i]))
            ied = int(np.sum(orbsi[0:i+1]))
            jst = int(np.sum(orbsj[0:j]))
            jed = int(np.sum(orbsj[0:j+1]))
            hk[ist:ied,jst:jed] += hij[ib] * np.exp(-1j * 2 * np.pi* np.dot(k,R))
        if conjtran:
            if norbsi != norbsj:
                print('Error with Hamr2Hamk, please check the input. orbsi == orbsj ?')
                exit()
            else:
                hk = hk + np.transpose(np.conjugate(hk))
                for i in range(norbsi):
                    hk[i,i] = hk[i,i] * 0.5
            
        return hk
    
    def HamilContReal(self, H00r, H01r, S00r, S01r, bond00, bond01, num_orbs):
        # H00 , H01 in real space. H00, H01, intra and inter PL HHamiltonian.
        self.H00r = H00r
        self.H01r = H01r
        self.S00r = S00r
        self.S01r = S01r
        self.bond00 = bond00
        self.bond01 = bond01
        self.num_orbs = num_orbs
        
    def HamilContKrecp(self,kp):
        # Trans to k space. usually the device is  big enough, Gamma only is ok.
        self.H00k = self.Hamr2Hamk(hij = self.H00r,kpoint = kp,\
                    bond = self.bond00, orbsi = self.num_orbs, orbsj = self.num_orbs, \
                                    conjtran = True)
        self.H01k = self.Hamr2Hamk(hij = self.H01r,kpoint = kp, \
                    bond = self.bond01, orbsi = self.num_orbs, orbsj = self.num_orbs,\
                                    conjtran = False)
        self.S00k = self.Hamr2Hamk(hij = self.S00r,kpoint = kp, \
                    bond = self.bond00, orbsi = self.num_orbs, orbsj = self.num_orbs,\
                                    conjtran = True)
        self.S01k = self.Hamr2Hamk(hij = self.S01r,kpoint = kp, \
                    bond = self.bond01, orbsi = self.num_orbs, orbsj = self.num_orbs,\
                                    conjtran = False)
    
    def SurfGFE(self,E):
        Ene = E
        bulkgf, surfgf_top, surfgf_bot = \
                    self.iterationGF(self.H00k, self.H01k, self.S00k,self.S01k, Ene,\
                        self.Eta, self.max_iteration, self.epsilon)
        
        return np.asarray([bulkgf, surfgf_top, surfgf_bot])
        
    def iterationGF(self, H00, H01, S00, S01, E, eta, max_iteration=100, epsilon = 1.0E-6):
        if H00.shape != H01.shape or H00.shape != S00.shape and S00.shape  != S01.shape:
            print('The size of H00 H01 and S00 S01 must be the same! please check!')
            exit()
        # initial for iteration.
        E = E + 1.0j *eta   # 加入虚部，推迟格林函数。
        ep0    = E*S00 - H00
        eps0   = np.copy(ep0)
        eps0_i = np.copy(ep0)
        alpha0 = -1*(E*S01 - H01)
        beta0  = -1*np.conj(np.transpose((E*S01 - H01)))
        iflag = 0
        for it in range(max_iteration):
            gf = np.linalg.inv(ep0)
            A  = np.dot(alpha0,gf)
            B  = np.dot(beta0,gf)
            alpha1 = np.dot(A,alpha0)
            beta1  = np.dot(B,beta0)
                                    
            agb  = np.dot(A,beta0)
            bga  = np.dot(B,alpha0)
            
            ep1    = ep0 - agb - bga
            eps1   = eps0 - agb
            eps1_i = eps0_i - bga
            ep0    = np.copy(ep1)
            eps0   = np.copy(eps1)
            eps0_i = np.copy(eps1_i)
            alpha0 = np.copy(alpha1)
            beta0  = np.copy(beta1)             
            
            if np.sum(np.abs(alpha0))< epsilon \
                and np.sum(np.abs(beta0)) < epsilon:
                iflag +=1 
            if iflag>2:
                ## 收敛，计算表面格林函数
                bulkgf     = np.linalg.inv(ep0)
                surfgf_top = np.linalg.inv(eps0)   # 上表面
                surfgf_bot = np.linalg.inv(eps0_i) # 下表面
                break
            elif it == max_iteration-1:
                print('convergence is not obtained within %d steps' %max_iteration)
                exit()
        return bulkgf, surfgf_top, surfgf_bot
    

    

    


class SelfEnergy(SurfGF):
    def __init__(self,paras):
        print('# init SelfEnergy calss.')
        super(SelfEnergy, self).__init__(paras)
    
    def HamilSCReal(self, Hsc, Ssc, bondsc, orbss, orbsc):
        self.Hsc    = Hsc
        self.Ssc    = Ssc
        self.bondsc = bondsc
        self.orbss  = orbss
        self.orbsc  = orbsc
        
    def HamilSCKrecp(self,kp):
        self.Hsck = self.Hamr2Hamk(hij = self.Hsc,kpoint = kp, \
                    bond = self.bondsc, orbsi = self.orbss, orbsj = self.orbsc,\
                                    conjtran = False)
        self.Ssck = self.Hamr2Hamk(hij = self.Ssc,kpoint = kp, \
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

class NEGFTrans(SelfEnergy):
    def __init__(self,paras):
        print('# init NEGFTrans calss.')
        super(NEGFTrans, self).__init__(paras) 
    
    def HamilSReal(self, Hss, Sss, bondss, orbss):
        self.Hss    = Hss
        self.Sss    = Sss
        self.bondss = bondss
        self.orbss  = orbss
    
    def HamilSKrecp(self,kp):
        self.Hssk = self.Hamr2Hamk(hij = self.Hss,kpoint = kp, \
                    bond = self.bondss, orbsi = self.orbss, orbsj = self.orbss,\
                                    conjtran = True)
        self.Sssk = self.Hamr2Hamk(hij = self.Sss,kpoint = kp, \
                    bond = self.bondss, orbsi = self.orbss, orbsj = self.orbss,\
                                    conjtran = True)
    
    def GetSelfEne(self, SigmaEs):
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


