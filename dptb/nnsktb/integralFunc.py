import  torch as th
from dptb.utils.constants import atomic_num_dict_r
from dptb.nnsktb.formula import SKFormula

# define the function for output all the hoppongs for given i,j.


class SKintHops(SKFormula):
    def __init__(self,mode='varTang96') -> None:
        super().__init__(mode='varTang96')

    def get_skhops(self, bonds, coeff_paras: dict, sk_bond_ind: dict):
        '''> The function `get_skhops` takes in a list of bonds, a dictionary of Slater-Koster coeffient parameters obtained in sknet fitting,
        and a dictionary of sk_bond_ind obtained in skintType func, and returns a list of Slater-Koster hopping integrals.
        
        Parameters
        ----------
        bonds
            the bond list, with the first 7 columns being the bond information, and the 8-th column being the
        bond length.
        coeff_paras : dict
            a dictionary of the coeffient parameters for each SK term.
        sk_bond_ind : dict
            a dictionary that contains the of `key/name` of the dict of Slater-Koster coeffient parameters for each bond type.
        
        Returns
        -------
            a list of hopping matrices.
        
        '''

        hoppings = []
        for ib in range(len(bonds)):
            ibond = bonds[ib,0:7].astype(int)
            rij = bonds[ib,7]
            ia, ja = atomic_num_dict_r[ibond[0]], atomic_num_dict_r[ibond[2]]           
            paraArray = th.stack([coeff_paras[isk] for isk in sk_bond_ind[f'{ia}-{ja}']])

            paras = {'paraArray':paraArray,'rij':rij}
            hij = self.skhij(**paras)
            hoppings.append(hij)

        return hoppings
