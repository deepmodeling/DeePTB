# define the integrals formula.
import torch
from abc import ABC, abstractmethod
from dptb.nn.sktb.bondlengthDB import bond_length_list

class BaseHopping(ABC):
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

class HoppingFormula(BaseHopping):

    def __init__(self, functype='varTang96',overlap=False) -> None:
        super(HoppingFormula, self).__init__()
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

        elif functype == 'NRL':
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
        elif self.functype == 'NRL':
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

        if self.functype == 'NRL':
            return self.NRL_OVERLAP(rij=rij, **kwargs)
        else:
            raise ValueError('No such formula')


    def varTang96(self, rij, paraArray, rcut:torch.Tensor = torch.tensor(6), w:torch.Tensor = 0.1, **kwargs):
        """> This function calculates the value of the variational form of Tang et al 1996. without the
        environment dependent

                $$ h(rij) = \alpha_1 * (rij)^(-\alpha_2) * exp(-\alpha_3 * (rij)^(\alpha_4))$$
        """
        if isinstance(paraArray, list):
            paraArray = torch.tensor(paraArray)
        assert len(paraArray.shape) in {2, 1}, 'paraArray should be a 2d tensor or 1d tensor'
        paraArray = paraArray.view(-1, self.num_paras)
        #alpha1, alpha2, alpha3, alpha4 = paraArray[:, 0], paraArray[:, 1]**2, paraArray[:, 2]**2, paraArray[:, 3]**2
        alpha1, alpha2, alpha3, alpha4 = paraArray[:, 0], paraArray[:, 1].abs(), paraArray[:, 2].abs(), paraArray[:, 3].abs()

        return alpha1 * rij**(-alpha2) * torch.exp(-alpha3 * rij**alpha4)/(1+torch.exp((rij-rcut)/w))

    def powerlaw(self, rij, paraArray, r0:torch.Tensor, rcut:torch.Tensor = torch.tensor(6), w:torch.Tensor = 0.1, **kwargs):
        """> This function calculates the value of the variational form of Tang et al 1996. without the
        environment dependent

                $$ h(rij) = \alpha_1 * (rij / r_ij0)^(\lambda + \alpha_2)
        """
        if isinstance(paraArray, list):
            paraArray = torch.tensor(paraArray)
        assert len(paraArray.shape) in {2, 1}, 'paraArray should be a 2d tensor or 1d tensor'
        
        paraArray = paraArray.view(-1, self.num_paras)
        #alpha1, alpha2, alpha3, alpha4 = paraArray[:, 0], paraArray[:, 1]**2, paraArray[:, 2]**2, paraArray[:, 3]**2
        alpha1, alpha2 = paraArray[:, 0], paraArray[:, 1].abs()

        r0 = r0 / 1.8897259886
        return alpha1 * (r0/rij)**(1 + alpha2) / (1+torch.exp((rij-rcut)/w))

    def NRL_HOP(self, rij, paraArray, rcut:torch.Tensor = torch.tensor(6), w:torch.Tensor = 0.1, **kwargs):
        """
        This function calculates the SK integral value of the form of NRL-TB 

            H_{ll'u} = (a + b R + c R^2)exp(-d^2 R) f(R)
            a,b,c,d are the parameters, R is r_ij

            f(r_ij) = [1+exp((r_ij-rcut+5w)/w)]^-1;    (r_ij <  rcut)
                    = 0;                               (r_ij >= rcut)

        """
        if isinstance(paraArray, list):
            paraArray = torch.tensor(paraArray)
        assert len(paraArray.shape) in {2, 1}, 'paraArray should be a 2d tensor or 1d tensor'
        
        paraArray = paraArray.view(-1, self.num_paras)
        a, b, c, d = paraArray[:, 0], paraArray[:, 1], paraArray[:, 2], paraArray[:, 3]

        f_rij = 1/(1+torch.exp((rij-rcut+5*w)/w))
        f_rij[rij>=rcut] = 0.0

        return (a + b * rij + c * rij**2) * torch.exp(-d**2 * rij)*f_rij

    def NRL_OVERLAP(self, rij, paraArray, paraconst, rcut:torch.float32 = torch.tensor(6), w:torch.float32 = 0.1, **kwargs):
        """
        This function calculates the Overlap value of the form of NRL-TB 

            S_{ll'u} = (delta_ll' + a R + b R^2 + c R^3)exp(-d^2 R) f(R)
            a,b,c,d are the parameters, R is r_ij

            f(r_ij) = [1+exp((r_ij-rcut+5w)/w)]^-1;    (r_ij <  rcut)
                    = 0;                               (r_ij >= rcut)
        # delta
        """
        if isinstance(paraArray, list):
            paraArray = torch.tensor(paraArray)
        if isinstance(paraconst, list):
            paraconst = torch.tensor(paraconst)

        assert len(paraArray.shape) in {2, 1}, 'paraArray should be a 2d tensor or 1d tensor'
        assert paraconst is not None, 'paraconst should not be None'
        assert len(paraconst.shape) in {2, 1}, 'paraconst should be a 2d tensor or 1d tensor'
        
        paraArray = paraArray.view(-1, self.num_paras)
        paraconst = paraconst.view(-1, 1)

        a, b, c, d = paraArray[:, 0], paraArray[:, 1], paraArray[:, 2], paraArray[:, 3]
        delta_ll = paraconst[:,0]

        f_rij = 1/(1+torch.exp((rij-rcut+5*w)/w))
        f_rij[rij>=rcut] = 0.0

        return (delta_ll + a * rij + b * rij**2 + c * rij**3) * torch.exp(-d**2 * rij)*f_rij
    
class SKhopping(HoppingFormula):
    def __init__(self, functype="varTang96", overlap=False) -> None:
        super(SKhopping, self).__init__(functype=functype, overlap=overlap)

    def get_skhops(self, edge_anumber, rij: torch.Tensor, params: torch.Tensor, rcut:torch.Tensor = torch.tensor(6.), w:torch.Tensor = torch.tensor(0.1)):
        '''> The function `get_skhops` takes in a list of bonds, a dictionary of Slater-Koster coeffient parameters obtained in sknet fitting,
        and a dictionary of sk_bond_ind obtained in skintType func, and returns a list of Slater-Koster hopping integrals.
        
        Parameters
        ----------
        edge_anumber: torch.Tensor
            the bond type tensor, shaped [2,N], [[i_atomic_number], [j_atomic_number]]
        rij: torch.Tensor
            bond_length, shaped torch.tensor(N)
        edge_index: torch.Tensor
            the bond index tensor, shaped [2,N], [[i_atom], [j_atom]]
        params: torch.Tensor
            Tensor containing sk hopping parameters, shaped [N, n_orb, n_formula]
        
        Returns
        -------
        hij: torch.Tensor
            a Tensor of hopping SK integrals, shaped [N, n_orb]
        
        ''' 
        r0 = 0.5*(bond_length_list[edge_anumber[0]] +  bond_length_list[edge_anumber[1]])
        N, n_orb, n_formula = params.shape
        hij = self.skhij(
            paraArray=params.reshape(-1, n_formula), 
            rij=rij.unsqueeze(1).repeat(1, n_orb).reshape(-1), 
            r0=r0.unsqueeze(1).repeat(1, n_orb).reshape(-1),
            rcut=rcut, 
            w=w
            ) # shaped (N * n_orb)

        return hij.reshape(N, n_orb)
    
    def get_skoverlaps(self, rij: torch.Tensor, params: torch.Tensor, const: torch.Tensor, rcut: torch.Tensor = torch.tensor(6.), w:torch.Tensor = torch.tensor(0.1)):
        """ The function `get_skoverlaps` takes in a list of bonds, a dictionary of Slater-Koster coeffient parameters obtained in sknet fitting,
        and a dictionary of sk_bond_ind obtained in skintType func, and returns a list of Slater-Koster hopping integrals.

        Parameters
        ----------
        bonds
            the bond list, with the first 7 columns being the bond information, and the 8-th column being the
        bond length.
        coeff_paras : dict
            a dictionary of the coeffient parameters for each SK term.
        bond_index_dict : dict
            a dictionary that contains the of `key/name` of the dict of Slater-Koster coeffient parameters for each bond type.

        Returns
        -------
            a list of overlap SK integrals.
        """



        N, n_orb, n_formula = params.shape
        sij = self.sksij(
            params=params.reshape(-1, n_formula),
            rij=rij.unsqueeze(1).repeat(1, n_orb).reshape(-1),
            const=const.reshape(-1),
            rcut=rcut, 
            w=w
            )

        return sij.reshape(N, n_orb)