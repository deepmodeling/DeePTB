import  torch as th
from dptb.utils.constants import atomic_num_dict_r
from dptb.nnsktb.formula import SKFormula
from dptb.utils.index_mapping import Index_Mapings
from dptb.nnsktb.skintTypes import all_skint_types, all_onsite_intgrl_types, NRL_skint_type_constants
import re

# define the function for output all the hoppongs for given i,j.


class SKintHops(SKFormula):
    def __init__(self, proj_atom_anglr_m, atomtype=None, mode='hopping', functype='varTang96',overlap=False) -> None:
        super().__init__(functype=functype,overlap=overlap)
        IndMap = Index_Mapings()
        IndMap.update(proj_atom_anglr_m=proj_atom_anglr_m)
        bond_index_map, _ = IndMap.Bond_Ind_Mapings()
        if mode == 'hopping':
            # _, _, sk_bond_ind_dict = all_skint_types(bond_index_map)
            _, reducted_skint_types, sk_bond_ind_dict  = all_skint_types(bond_index_map)
            self.bond_index_dict = sk_bond_ind_dict
            self.para_Consts = None 
            
            if re.search("NRL",functype): 
                self.para_Consts = NRL_skint_type_constants(reducted_skint_types)
                # call to get the para constants!
        # special onsite mode for strain, which use same sk strategy as hopping.
        elif mode == 'onsite':
            onsite_strain_index_map, _,  _, _ = IndMap.Onsite_Ind_Mapings(onsitemode='strain', atomtype=atomtype)
            _, _, onsite_strain_ind_dict = all_onsite_intgrl_types(onsite_strain_index_map)
            self.bond_index_dict = onsite_strain_ind_dict

        else:
            raise ValueError("Unknown mode '%s' for SKintHops" %mode)


    def get_skhops(self, batch_bonds, coeff_paras: dict, rcut:th.float32 = th.tensor(6), w:th.float32 = 0.1):
        '''> The function `get_skhops` takes in a list of bonds, a dictionary of Slater-Koster coeffient parameters obtained in sknet fitting,
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
            a list of hopping SK integrals.
        
        ''' 
        # TODO: 可能得优化目标：能不能一次性把所有的rij 计算出来。而不是循环计算每一个bond.
        batch_hoppings = {}
        for fi in batch_bonds.keys():
            hoppings = []
            for ib in range(len(batch_bonds[fi])):
                ibond = batch_bonds[fi][ib,1:8]
                rij = batch_bonds[fi][ib,8]
                ia, ja = atomic_num_dict_r[int(ibond[0])], atomic_num_dict_r[int(ibond[2])]
                # take all the coeffient parameters for the bond type.
                paraArray = th.stack([coeff_paras[isk] for isk in self.bond_index_dict[f'{ia}-{ja}']])

                paras = {'paraArray':paraArray,'rij':rij, 'iatomtype':ia, 'jatomtype':ja, 'rcut':rcut,'w':w}
                hij = self.skhij(**paras)
                hoppings.append(hij)
            batch_hoppings.update({fi:hoppings})

        return batch_hoppings
    
    def get_skoverlaps(self, batch_bonds, coeff_paras: dict, rcut:th.float32 = th.tensor(6), w:th.float32 = 0.1):
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
        batch_overlaps = {}
        for fi in batch_bonds.keys():
            overlaps = []
            for ib in range(len(batch_bonds[fi])):
                ibond = batch_bonds[fi][ib,1:8]
                rij = batch_bonds[fi][ib,8]
                ia, ja = atomic_num_dict_r[int(ibond[0])], atomic_num_dict_r[int(ibond[2])]
                # take all the coeffient parameters for the bond type.
                paraArray = th.stack([coeff_paras[isk] for isk in self.bond_index_dict[f'{ia}-{ja}']])

                if self.para_Consts is not None:
                    paraconst = th.stack([self.para_Consts[isk] for isk in self.bond_index_dict[f'{ia}-{ja}']])
                else:
                    paraconst = None

                paras = {'paraArray':paraArray,'paraconst':paraconst, 'rij':rij, 'iatomtype':ia, 'jatomtype':ja, 'rcut':rcut,'w':w}
                sij = self.sksij(**paras)
                overlaps.append(sij)
            batch_overlaps.update({fi:overlaps})

        return batch_overlaps


    