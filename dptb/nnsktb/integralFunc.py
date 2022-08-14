import  torch as th
from dptb.utils.constants import atomic_num_dict_r
from dptb.nnsktb.formula import SKFormula

# define the function for output all the hoppongs for given i,j.


class SKintHops(SKFormula):
    def __init__(self,mode='varTang96') -> None:
        super().__init__(mode='varTang96')

    def get_skhops(self, bonds, coeff_paras: dict, bond_ind: dict):
        hoppings = []
        for ib in range(len(bonds)):
            ibond = bonds[ib,0:7].astype(int)
            rij = bonds[ib,7]
            ia, ja = atomic_num_dict_r[ibond[0]], atomic_num_dict_r[ibond[2]]           
            paraArray = th.stack([coeff_paras[isk] for isk in bond_ind[f'{ia}-{ja}']])

            paras = {'paraArray':paraArray,'rij':rij}
            hij = self.skhij(**paras)
            hoppings.append(hij)

        return hoppings
