# define the integrals formula.
import torch as th
from abc import ABC, abstractmethod
from dptb.nnsktb.bondlengthDB import bond_length
import re

class BaseSK(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def skhij(self, rij, **kwargs):
        '''This is a wrap function for a self-defined formula of sk integrals. one can easily modify it into whatever form they want.
        
        Returns
        -------
            The function defined by type is called to cal skhij and returned.
        
        '''
        pass

class SKFormula(BaseSK):

    def __init__(self, functype='varTang96',overlap=False) -> None:
        super(SKFormula, self).__init__()
        # one can modify this by add his own formula with the name functype to deifine num of pars.
        self.overlap = overlap
        if functype == 'varTang96':
            self.functype = functype
            self.num_paras = 4
            assert hasattr(self, 'varTang96')
       
        elif functype == 'powerlaw':
            self.functype = functype
            self.num_paras = 2
            assert hasattr(self, 'powerlaw')

        elif re.search("NRL",functype):
            self.functype = functype
            self.num_paras = 4
            assert hasattr(self, 'NRL_HOP')
            if overlap:
                self.overlap_num_paras = 4
                assert hasattr(self, 'NRL_OVERLAP')

        elif functype =='custom':
             # the functype custom, is for user to define their own formula.
            # just modify custom to the name of your formula.
            # and define the funnction self.custom(rij, paraArray, **kwargs)
            self.functype = functype
            self.num_paras = None # defined by custom.
            assert hasattr(self, 'custom')
        else:
            raise ValueError('No such formula')
        

    def skhij(self, rij, **kwargs):
        '''This is a wrap function for a self-defined formula of sk integrals. one can easily modify it into whatever form they want.
        
        Returns
        -------
            The function defined by functype is called to cal skhij and returned.
        
        '''

        if self.functype == 'varTang96':
            return self.varTang96(rij=rij, **kwargs)
        elif self.functype == 'powerlaw':
            return self.powerlaw(rij=rij, **kwargs)
        elif re.search("NRL",self.functype):
            return self.NRL_HOP(rij=rij, **kwargs)
        else:
            raise ValueError('No such formula')

    def sksij(self,rij,**kwargs):
        '''This is a wrap function for a self-defined formula of sk overlap. one can easily modify it into whatever form they want.
        
        Returns
        -------
            The function defined by functype is called to cal sk sij and returned.
        
        '''
        assert self.overlap, 'overlap is False, no overlap function is defined.'

        if self.functype == 'NRLv0':
            return self.NRL_OVERLAP(rij=rij,overlap_type=0, **kwargs)
        elif self.functype == 'NRLv1':
            return self.NRL_OVERLAP(rij=rij,overlap_type=1, **kwargs)
        elif self.functype == "NRL":
            # default is designed for overlap type 1.
            return self.NRL_OVERLAP(rij=rij,overlap_type=1, **kwargs)
        else:
            raise ValueError('No such formula')


    def varTang96(self, rij, paraArray, rcut:th.float32 = th.tensor(6), w:th.float32 = 0.1, **kwargs):
        """> This function calculates the value of the variational form of Tang et al 1996. without the
        environment dependent

                $$ h(rij) = \alpha_1 * (rij)^(-\alpha_2) * exp(-\alpha_3 * (rij)^(\alpha_4))$$
        """
        if isinstance(paraArray, list):
            paraArray = th.tensor(paraArray)
        assert len(paraArray.shape) in {2, 1}, 'paraArray should be a 2d tensor or 1d tensor'
        paraArray = paraArray.view(-1, self.num_paras)
        #alpha1, alpha2, alpha3, alpha4 = paraArray[:, 0], paraArray[:, 1]**2, paraArray[:, 2]**2, paraArray[:, 3]**2
        alpha1, alpha2, alpha3, alpha4 = paraArray[:, 0], paraArray[:, 1].abs(), paraArray[:, 2].abs(), paraArray[:, 3].abs()

        return alpha1 * rij**(-alpha2) * th.exp(-alpha3 * rij**alpha4)/(1+th.exp((rij-rcut)/w))

    def powerlaw(self, rij, paraArray, iatomtype, jatomtype, rcut:th.float32 = th.tensor(6), w:th.float32 = 0.1, **kwargs):
        """> This function calculates the value of the variational form of Tang et al 1996. without the
        environment dependent

                $$ h(rij) = \alpha_1 * (rij / r_ij0)^(\lambda + \alpha_2)
        """
        if isinstance(paraArray, list):
            paraArray = th.tensor(paraArray)
        assert len(paraArray.shape) in {2, 1}, 'paraArray should be a 2d tensor or 1d tensor'
        
        paraArray = paraArray.view(-1, self.num_paras)
        #alpha1, alpha2, alpha3, alpha4 = paraArray[:, 0], paraArray[:, 1]**2, paraArray[:, 2]**2, paraArray[:, 3]**2
        alpha1, alpha2 = paraArray[:, 0], paraArray[:, 1].abs()

        # r0 = map(lambda x:(bond_length[iatomtype[x]]+bond_length[jatomtype[x]])/(2*1.8897259886), range(len(iatomtype)))
        # r0 = th.tensor(list(r0))
        r0 = (bond_length[iatomtype]+bond_length[jatomtype])/(2*1.8897259886)
        return alpha1 * (r0/rij)**(1 + alpha2) / (1+th.exp((rij-rcut)/w))

    def NRL_HOP(self, rij, paraArray, rcut:th.float32 = th.tensor(6), w:th.float32 = 0.1, **kwargs):
        """
        This function calculates the SK integral value of the form of NRL-TB 

            H_{ll'u} = (a + b R + c R^2)exp(-d^2 R) f(R)
            a,b,c,d are the parameters, R is r_ij

            f(r_ij) = [1+exp((r_ij-rcut+5w)/w)]^-1;    (r_ij <  rcut)
                    = 0;                               (r_ij >= rcut)

        """
        if isinstance(paraArray, list):
            paraArray = th.tensor(paraArray)
        assert len(paraArray.shape) in {2, 1}, 'paraArray should be a 2d tensor or 1d tensor'
        
        paraArray = paraArray.view(-1, self.num_paras)
        a, b, c, d = paraArray[:, 0], paraArray[:, 1], paraArray[:, 2], paraArray[:, 3]

        f_rij = 1/(1+th.exp((rij-rcut+5*w)/w))
        f_rij[rij>=rcut] = 0.0

        return (a + b * rij + c * rij**2) * th.exp(-d**2 * rij)*f_rij

    def NRL_OVERLAP(self, rij, paraArray, paraconst, rcut:th.float32 = th.tensor(6), w:th.float32 = 0.1, overlap_type=1, **kwargs):
        """
        This function calculates the Overlap value of the form of NRL-TB 

            S_{ll'u} = (delta_ll' + a R + b R^2 + c R^3)exp(-d^2 R) f(R)
            a,b,c,d are the parameters, R is r_ij

            f(r_ij) = [1+exp((r_ij-rcut+5w)/w)]^-1;    (r_ij <  rcut)
                    = 0;                               (r_ij >= rcut)
        # delta
        """
        if isinstance(paraArray, list):
            paraArray = th.tensor(paraArray)
        if isinstance(paraconst, list):
            paraconst = th.tensor(paraconst)

        assert len(paraArray.shape) in {2, 1}, 'paraArray should be a 2d tensor or 1d tensor'
        assert paraconst is not None, 'paraconst should not be None'
        assert len(paraconst.shape) in {2, 1}, 'paraconst should be a 2d tensor or 1d tensor'
        
        paraArray = paraArray.view(-1, self.num_paras)
        paraconst = paraconst.view(-1, 1)

        a, b, c, d = paraArray[:, 0], paraArray[:, 1], paraArray[:, 2], paraArray[:, 3]
        delta_ll = paraconst[:,0]

        f_rij = 1/(1+th.exp((rij-rcut+5*w)/w))
        f_rij[rij>=rcut] = 0.0

        if overlap_type==1:
            return (delta_ll + a * rij + b * rij**2 + c * rij**3) * th.exp(-d**2 * rij)*f_rij
        elif overlap_type ==0:
            return (a + b * rij + c * rij**2) * th.exp(-d**2 * rij)*f_rij
        else:
            raise ValueError('No such overlap type')