import torch  as th
import numpy as np
import re
from dptb.utils.constants import SKAnglrMHSID, au2Ang, anglrMId, NumHvals

class SKHSLists(object):
    """ This module is to build the Hamiltonian from the SK-type bond integral.
    """
    def __init__(self, skint, dtype='tensor') -> None:
        self.dtype = dtype
        self.hoppings = []
        self.onsiteEs = []
        self.overlaps = []
        self.onsiteSs = []
        self.use_orthogonal_basis = False
        self.hamil_blocks = None
        self.overlap_blocks = None
        self.SKInt = skint

    def update_struct(self, struct):
        self.__struct__ = struct
    
    def get_HS_list(self, bonds_onsite = None, bonds_hoppings=None):
        if bonds_onsite is None:
            bonds_onsite = self.__struct__.__bonds_onsite__
        assert len(bonds_onsite) == len(self.__struct__.__bonds_onsite__) 
        if bonds_hoppings is None:
            bonds_hoppings = self.__struct__.__bonds__ 
        assert len(bonds_hoppings) == len(self.__struct__.__bonds__)
        
        self.hoppings = []
        self.onsiteEs = []
        self.overlaps = []
        self.onsiteSs = []

        for ib in range(len(bonds_onsite)):
            ibond = bonds_onsite[ib].int()
            iatype = self.__struct__.proj_atom_symbols[ibond[1]]
            jatype = self.__struct__.proj_atom_symbols[ibond[3]]
            assert iatype == jatype, "i type should equal j type."
            num_onsite = self.__struct__.onsite_num[iatype]

            siteE = np.zeros([num_onsite])
            siteS = np.zeros([num_onsite])
            for ish in self.__struct__.proj_atom_anglr_m[iatype]:  # ['s','p',..]
                ishsymbol = ''.join(re.findall(r'[A-Za-z]',ish))
                shidi = anglrMId[ishsymbol]  # 0,1,2,...
                indx = self.__struct__.onsite_index_map[iatype][ish]
                siteE[indx] = self.SKInt.SiteE[iatype][shidi]
                siteS[indx] = 1.0
            if self.dtype == 'tensor':
                self.onsiteEs.append(th.from_numpy(siteE).float())
                self.onsiteSs.append(th.from_numpy(siteS).float())
            else:
                self.onsiteEs.append(siteE)
                self.onsiteSs.append(siteS)

        for ib in range(len(bonds_hoppings)):

            ibond = bonds_hoppings[ib,0:7].int()
            #dirvec = (self.__struct__.projected_struct.positions[ibond[3]]
            #          - self.__struct__.projected_struct.positions[ibond[1]]
            #          + np.dot(ibond[4:], self.__struct__.projected_struct.cell))
            #dist = np.linalg.norm(dirvec)
            #dist = dist / au2Ang  # the sk files is written in atomic unit
            dist = bonds_hoppings[ib,7].float() / au2Ang

            iatype = self.__struct__.proj_atom_symbols[ibond[1]]
            jatype = self.__struct__.proj_atom_symbols[ibond[3]]
            HKinterp12 = self.SKInt.sk_integral(itype=iatype, jtype=jatype, dist=dist)
            """
            for a A-B bond, there are two sk files, A-B and B-A.
            e.g.:
                sp hopping, A-B: sp. means A(s) - B(p) hopping. 
                we know, A(s) - B(p) => B(p)-A(s)
                therefore, from A-B sp, we know B-A ps.
            """

            if iatype == jatype:
                # view, the same addr. in mem.
                HKinterp21 = HKinterp12
            else:
                HKinterp21 = self.SKInt.sk_integral(itype=jatype, jtype=iatype, dist=dist)

            num_hops = self.__struct__.bond_num_hops[iatype + '-' + jatype]
            bondname = iatype + '-' + jatype
            hoppings = np.zeros([num_hops])
            overlaps = np.zeros([num_hops])

            for ish in self.__struct__.proj_atom_anglr_m[iatype]:
                ishsymbol = ''.join(re.findall(r'[A-Za-z]',ish))
                shidi = anglrMId[ishsymbol]
                # norbi = 2*shidi+1

                for jsh in self.__struct__.proj_atom_anglr_m[jatype]:
                    jshsymbol = ''.join(re.findall(r'[A-Za-z]',jsh))
                    shidj = anglrMId[jshsymbol]
                    # norbj = 2 * shidj + 1

                    if shidi < shidj:
                        Hvaltmp = HKinterp12[SKAnglrMHSID[ishsymbol + jshsymbol]]
                        Svaltmp = HKinterp12[SKAnglrMHSID[ishsymbol + jshsymbol] + NumHvals]
                    else:
                        Hvaltmp = HKinterp21[SKAnglrMHSID[ishsymbol + jshsymbol]]
                        Svaltmp = HKinterp21[SKAnglrMHSID[ishsymbol + jshsymbol] + NumHvals]
                    # print(iatype, jatype, self.ProjAnglrM,ish,jsh)
                    # print(bondname, self.__struct__.bond_index_map[bondname])
                    indx = self.__struct__.bond_index_map[bondname][ish +'-'+ jsh]
                    hoppings[indx] = Hvaltmp
                    overlaps[indx] = Svaltmp
            if self.dtype == 'tensor':
                self.hoppings.append(th.from_numpy(hoppings).float())
                self.overlaps.append(th.from_numpy(overlaps).float())
            else:
                self.hoppings.append(hoppings)
                self.overlaps.append(overlaps)