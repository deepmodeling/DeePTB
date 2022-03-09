import numpy as np
import warnings

class SurfGF(object):
    """ To calculate surface green's fucntion.

    Input
    -----
    eta: a small imaginary value for retarded GF.
    epsilon: the threshold of convergence for iteration process
    max_iteration: the max_iteration steps.
    """

    def __init__(self,
                 eta=0.001,
                 epsilon=1.0E-6,
                 max_iteration=100):

        self.max_iteration = max_iteration
        self.epsilon = epsilon
        self.eta = eta


    def get_sgf(self,
                omega,
                kpoint,
                H00, 
                H01, 
                S00, 
                S01, 
                bond00, 
                bond01, 
                num_orbs):

        """ calculate surface green function G(k,\omega).

        Input
        ----- 
        omega: quasi-particle energy w.r.t. Fermi level.
        Return
        ------ 
        np.asarray([bulkgf, surfgf_top, surfgf_bot])
        bulk, top surface and bottom surface green's function at k , \omega
       
        """

        # Trans to k space. 
        H00k = self.Hamr2Hamk(hij = H00, kpoint = kpoint,\
                    bond = bond00, orbsi = num_orbs, orbsj = num_orbs, conjtran = True)
        H01k = self.Hamr2Hamk(hij = H01, kpoint = kpoint, \
                    bond = bond01, orbsi = num_orbs, orbsj = num_orbs, conjtran = False)
        S00k = self.Hamr2Hamk(hij = S00, kpoint = kpoint, \
                    bond = bond00, orbsi = num_orbs, orbsj = num_orbs, conjtran = True)
        S01k = self.Hamr2Hamk(hij = S01, kpoint = kpoint, \
                    bond = bond01, orbsi = num_orbs, orbsj = num_orbs, conjtran = False)

        bulkgf, surfgf_top, surfgf_bot = self.iterationGF(H00k, H01k, S00k,S01k, omega, 
                                            self.eta, self.max_iteration, self.epsilon)
        
        return np.asarray([bulkgf, surfgf_top, surfgf_bot])


    @staticmethod
    def Hamr2Hamk(hij, kpoint, bond, orbsi, orbsj, conjtran = False):
        """ Transfer the Hamiltonian for real space to k space.
        
        Inputs
        ------
        hij: Hamiltonian or overlap blocks.
        kpoint: a single k-point
        bond: bond list
        orbsi: atom norbs for <i>s.
        orbsj: atom norbs for <j>s.

        Returns
        -------
        hk: Hamiltonian in k-space (complex matrix) with dimension [sum(orbsi) , sum(orbsj)].
        """

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
            # hk[ist:ied,jst:jed] += hij[ib] * np.exp(-1j * 2 * np.pi* np.dot(k,R))
            hk[ist:ied,jst:jed] +=  np.array(hij[ib],dtype=float) * np.exp(-1j * 2 * np.pi* np.dot(k,R))

        if conjtran:
            if norbsi != norbsj:
                print('Error with Hamr2Hamk, please check the input. orbsi == orbsj ?')
                exit()
            else:
                hk = hk + np.transpose(np.conjugate(hk))
                for i in range(norbsi):
                    hk[i,i] = hk[i,i] * 0.5
            
        return hk


    @staticmethod
    def iterationGF(H00, 
                    H01, 
                    S00, 
                    S01, 
                    E=0, 
                    eta=0.001, 
                    max_iteration = 100, 
                    epsilon = 1.0E-6):
        """ for Hamiltonian at given k to calcualte surface green's fucntion at given energy E.
        
        Input
        ----- 
        H00: 
        H01:
        S00:
        S01:
        E: 
        eta: 
        max_iteration: 
        epsilon:

        Return
        ------ 
        bulkgf: bulk green function
        surfgf_top: top surface green function
        surfgf_bot: bottom surface green function
        """

        if H00.shape != H01.shape or H00.shape != S00.shape and S00.shape  != S01.shape:
            raise ValueError('The size of H00: %s H01: %s should equal.' %(list(H00.shape), list(H01.shape)))
        if S00.shape != S01.shape:
            raise ValueError('The size of S00: %s S01: %s should equal.' %(list(S00.shape), list(S01.shape)))
        if H00.shape != S00.shape:
            raise ValueError('The size of H00: %s S00: %s should equal.' %(list(H00.shape), list(S00.shape)))

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
    