# define the integrals formula.
import torch
from abc import ABC, abstractmethod
from dptb.nn.sktb.bondlengthDB import bond_length_list

class BaseHopping(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_skhij(self, rij, **kwargs):
        '''This is a wrap function for a self-defined formula of sk integrals. one can easily modify it into whatever form they want.
        
        Returns
        -------
            The function defined by type is called to cal skhij and returned.
        
        '''
        pass

class HoppingFormula(BaseHopping):
    num_paras_dict = {
        'varTang96': 4,
        'powerlaw': 2,
        'NRL': 4,
        'custom': None,
    }

    def __init__(self, functype='varTang96',overlap=False) -> None:
        super(HoppingFormula, self).__init__()
        # one can modify this by add his own formula with the name functype to deifine num of pars.
        self.overlap = overlap
        if functype == 'varTang96':
            assert hasattr(self, 'varTang96')
       
        elif functype == 'powerlaw':
            assert hasattr(self, 'powerlaw')

        elif functype == 'NRL':
            assert hasattr(self, 'NRL_HOP')
            if overlap:
                assert hasattr(self, 'NRL_OVERLAP')

        elif functype =='custom':
             # the functype custom, is for user to define their own formula.
            # just modify custom to the name of your formula.
            # and define the funnction self.custom(rij, paraArray, **kwargs)
            assert hasattr(self, 'custom')
        else:
            raise ValueError('No such formula')
        
        self.functype = functype
        self.num_paras = self.num_paras_dict[functype]
        

    def get_skhij(self, rij, **kwargs):
        '''This is a wrap function for a self-defined formula of sk integrals. one can easily modify it into whatever form they want.
        
        Returns
        -------
            The function defined by functype is called to cal skhij and returned.
        
        '''

        if self.functype == 'varTang96':
            return self.varTang96(rij=rij, **kwargs)
        elif self.functype == 'powerlaw':
            return self.powerlaw(rij=rij, **kwargs)
        elif self.functype == 'NRL':
            return self.NRL_HOP(rij=rij, **kwargs)
        else:
            raise ValueError('No such formula')

    def get_sksij(self,rij,**kwargs):
        '''This is a wrap function for a self-defined formula of sk overlap. one can easily modify it into whatever form they want.
        
        Returns
        -------
            The function defined by functype is called to cal sk sij and returned.
        
        '''
        assert self.overlap, 'overlap is False, no overlap function is defined.'

        if self.functype == 'NRL':
            return self.NRL_OVERLAP(rij=rij, **kwargs)
        elif self.functype == "powerlaw":
            return self.powerlaw(rij=rij, **kwargs)
        elif self.functype == "varTang96":
            return self.varTang96(rij=rij, **kwargs)
        else:
            raise ValueError('No such formula')


    def varTang96(self, rij: torch.Tensor, paraArray: torch.Tensor, rs:torch.Tensor = torch.tensor(6), w:torch.Tensor = 0.1, **kwargs):
        """> This function calculates the value of the variational form of Tang et al 1996. without the
        environment dependent

                $$ h(rij) = \alpha_1 * (rij)^(-\alpha_2) * exp(-\alpha_3 * (rij)^(\alpha_4))$$

        Parameters
        ----------
        rij : torch.Tensor([N, 1]/[N])
            the bond length vector, have the same length of the bond index vector.
        paraArray : torch.Tensor([N, ..., 4])
            The parameters for computing varTang96's type hopping integrals, the first dimension should have the 
            same length of the bond index vector, while the last dimenion if 4, which is the number of parameters
            for each varTang96's type formula.
        rcut : torch.Tensor, optional
            cut-off by half at which value, by default torch.tensor(6)
        w : torch.Tensor, optional
            the decay factor, the larger the smoother, by default 0.1

        Returns
        -------
        _type_
            _description_
        """
        
        rij = rij.reshape(-1)
        assert paraArray.shape[-1] == 4 and paraArray.shape[0] == len(rij), 'paraArray should be a 2d tensor with the last dimenion if 4, which is the number of parameters for each varTang96\'s type formula.'
        alpha1, alpha2, alpha3, alpha4 = paraArray[..., 0], paraArray[..., 1].abs(), paraArray[..., 2].abs(), paraArray[..., 3].abs()
        shape = [-1]+[1] * (len(alpha1.shape)-1)
        rij = rij.reshape(shape)
        return alpha1 * rij**(-alpha2) * torch.exp(-alpha3 * rij**alpha4)/(1+torch.exp((rij-rs)/w))

    def powerlaw(self, rij, paraArray, r0:torch.Tensor, rs:torch.Tensor = torch.tensor(6), w:torch.Tensor = 0.1, **kwargs):
        """> This function calculates the value of the variational form of Tang et al 1996. without the
        environment dependent

                $$ h(rij) = \alpha_1 * (rij / r_ij0)^(\lambda + \alpha_2)
        """

        #alpha1, alpha2, alpha3, alpha4 = paraArray[:, 0], paraArray[:, 1]**2, paraArray[:, 2]**2, paraArray[:, 3]**2
        alpha1, alpha2 = paraArray[..., 0], paraArray[..., 1].abs()
        shape = [-1]+[1] * (len(alpha1.shape)-1)
        rij = rij.reshape(shape)
        r0 = r0.reshape(shape)

        r0 = r0 / 1.8897259886
        return alpha1 * (r0/rij)**(1 + alpha2) / (1+torch.exp((rij-rs)/w))

    def NRL_HOP(self, rij, paraArray, rc:torch.Tensor = torch.tensor(6), w:torch.Tensor = 0.1, **kwargs):
        """
        This function calculates the SK integral value of the form of NRL-TB 

            H_{ll'u} = (a + b R + c R^2)exp(-d^2 R) f(R)
            a,b,c,d are the parameters, R is r_ij

            f(r_ij) = [1+exp((r_ij-rcut+5w)/w)]^-1;    (r_ij <  rcut)
                    = 0;                               (r_ij >= rcut)

        """
        rij = rij.reshape(-1)
        a, b, c, d = paraArray[..., 0], paraArray[..., 1], paraArray[..., 2], paraArray[..., 3]
        shape = [-1]+[1] * (len(a.shape)-1)
        rij = rij.reshape(shape)
        f_rij = 1/(1+torch.exp((rij-rc+5*w)/w))
        f_rij[rij>=rc] = 0.0

        return (a + b * rij + c * rij**2) * torch.exp(-d**2 * rij)*f_rij

    def NRL_OVERLAP(self, rij, paraArray, paraconst, rc:torch.float32 = torch.tensor(6), w:torch.float32 = 0.1, **kwargs):
        """
        This function calculates the Overlap value of the form of NRL-TB 

            S_{ll'u} = (delta_ll' + a R + b R^2 + c R^3)exp(-d^2 R) f(R)
            a,b,c,d are the parameters, R is r_ij

            f(r_ij) = [1+exp((r_ij-rcut+5w)/w)]^-1;    (r_ij <  rcut)
                    = 0;                               (r_ij >= rcut)
        # delta
        """

        assert paraArray.shape[:-1] == paraconst.shape, 'paraArray and paraconst should have the same shape except the last dimenion.'
        rij = rij.reshape(-1)
        assert len(rij) == len(paraArray), 'rij and paraArray should have the same length.'

        a, b, c, d = paraArray[..., 0], paraArray[..., 1], paraArray[..., 2], paraArray[..., 3]
        delta_ll = paraconst
        shape = [-1]+[1] * (len(a.shape)-1)
        rij = rij.reshape(shape)

        f_rij = 1/(1+torch.exp((rij-rc+5*w)/w))
        f_rij[rij>=rc] = 0.0

        return (delta_ll + a * rij + b * rij**2 + c * rij**3) * torch.exp(-d**2 * rij)*f_rij
    
    @classmethod
    def num_params(cls, funtype):
        return cls.num_paras_dict[funtype]
    