import  torch as th
from dptb.utils.constants import atomic_num_dict_r
from dptb.nnsktb.formula import SKFormula

# define the function for output all the hoppongs for given i,j.


class SKintHops(SKFormula):
    def __init__(self,mode='varTang96') -> None:
        super().__init__(mode=mode)

    def get_skhops(self, batch_bonds, coeff_paras: dict, sk_bond_ind: dict, rcut:th.float32 = th.tensor(6), w:th.float32 = 0.1):
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

        batch_hoppings = {}
        for ik in batch_bonds.keys():
            hoppings = []
            for ib in range(len(batch_bonds[ik])):
                ibond = batch_bonds[ik][ib,1:8]
                rij = batch_bonds[ik][ib,8]
                ia, ja = atomic_num_dict_r[int(ibond[0])], atomic_num_dict_r[int(ibond[2])]
                paraArray = th.stack([coeff_paras[isk] for isk in sk_bond_ind[f'{ia}-{ja}']])

                paras = {'paraArray':paraArray,'rij':rij,'rcut':rcut,'w':w}
                hij = self.skhij(**paras)
                hoppings.append(hij)
            batch_hoppings.update({ik:hoppings})

        return batch_hoppings
