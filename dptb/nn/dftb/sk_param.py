import os
import sys
import torch
from dptb.utils.tools import format_readline
from dptb.utils.constants import NumHvals,MaxShells
import logging
from typing import Tuple, Union, Dict
from dptb.data.transforms import OrbitalMapper
from dptb.utils._xitorch.interpolate import Interp1D

log = logging.getLogger(__name__)

class SKParam:
    # 键积分存储顺序
    intgl_order = {
        "s-s": slice(9,10,None),
        "s-p": slice(8,9,None),
        "s-d": slice(7,8,None),
        "p-p": slice(5,7,None),
        "p-d": slice(3,5,None),
        "d-d": slice(0,3,None)
    }
    support_full_basis = ['1s', '1p', '1d']
    onsite_orb_map = {'1s': 0, '1p': 1, '1d': 2}
    
    def __init__(self,
                basis: Dict[str, Union[str, list]]=None,
                idp_sk: Union[OrbitalMapper, None]=None,
                skdata: Union[str,dict] = None) -> None:

        if basis is not None:
            self.idp_sk = OrbitalMapper(basis, method="sktb")
            if idp_sk is not None:
                assert idp_sk.basis == self.idp_sk.basis, "The basis of idp and basis should be the same."
        else:
            assert idp_sk is not None, "Either basis or idp should be provided."
            self.idp_sk = idp_sk
        
        self.idp_sk.get_orbpair_maps()
        self.idp_sk.get_skonsite_maps()

        assert len(self.idp_sk.full_basis)<=3, "The dftb mode only supports 1s, 1p, 1d orbitals. maxmum of orbitals is 3." 
        for iorb in self.idp_sk.full_basis:
            assert iorb in self.support_full_basis, "The dftb mode only supports 1s, 1p, 1d orbitals."

        bond_types = self.idp_sk.bond_types

        assert skdata is not None, "You need to provide the skdata."

        if isinstance(skdata,str):        
            if '.' in skdata and skdata.split('.')[-1] == 'pth':
                log.info(f'Loading the skdict from the file: {skdata} ......')
                skdict = torch.load(skdata)
                for ikey in ['Distance', 'Hopping', 'Overlap', 'OnsiteE']:
                    assert ikey in skdict, f"The key: {ikey} is not in the skdict."
                
                for ibtype in bond_types:
                    for ikey in ['Distance', 'Hopping', 'Overlap']:
                        if ibtype not in skdict[ikey]:
                            log.error("The bond type: " + ibtype + " is not in the skdict-"+ikey)
                            raise ValueError("The bond type: " + ibtype + " is not in the skdict.")
                    if ibtype.split('-')[0] == ibtype.split('-')[1]:
                        if ibtype.split('-')[0] not in skdict['OnsiteE']:
                            log.error("The atom type: " + ibtype.split('-')[0] + " is not in the skdict['OnsiteE'].")
                            raise ValueError("The atom type: " + ibtype.split('-')[0] + " is not in the skdict['OnsiteE'].")
            else:
                log.info(f'Reading the skfiles from the path: {skdata} ......')
                skfiles = {}
                for ibtype in bond_types:
                    if not os.path.exists(skdata + '/' + ibtype + '.skf'):
                        log.error('Didn\'t find the skfile: ' + skdata + '/' + ibtype + '.skf')
                        raise FileNotFoundError('Didn\'t find the skfile: ' + skdata + '/' + ibtype + '.skf')
                    else:
                        skfiles[ibtype] = skdata + '/' + ibtype + '.skf'

                skdict = self.read_skfiles(skfiles)
        elif isinstance(skdata,dict):
            skdict = skdata
        else:
            log.error("The skdata should be a dict or string for a file path.")
            raise ValueError("The skdata should be a dict or string for a file path.")

        self.skdict = self.format_skparams(skdict)

    @classmethod
    def read_skfiles(self, skfiles):
        '''It reads the Slater-Koster files names and returns the grid distance, number of grids, 
        and the Slater-Koster integrals

        Parameters
        ----------
        skfiles
            a list of skfiles.

        Note: the Slater-Koster files name now consider that all the atoms are interacted with each other.
        Therefore, the number of Slater-Koster files is num_atom_types *2.

        Returns
        -------
        skdict:
            {
                "bond_type1": { 
                                "Distlist": torch.tensor, 
                                "Hintgrl": torch.tensor, 
                                "Sintgrl": torch.tensor,
                                "OnSiteE": torch.tensor[Es,Ep,Ed],
                                },
                ...
            }
        '''
        assert isinstance(skfiles, dict)
        skfiletypes = list(skfiles.keys())

        skdict = {}
        skdict['OnsiteE'] = {}
        skdict['Hopping']  = {}
        skdict['Overlap']  = {}
        skdict["Distance"] = {}
        for isktype in skfiletypes:
            filename = skfiles[isktype]
            log.info('Reading SlaterKoster File......')
            log.info(' ' + filename)
            fr = open(filename)
            data = fr.readlines()
            fr.close()
            # Line 1
            datline = format_readline(data[0])
            gridDist, ngrid = float(datline[0]), int(datline[1])
            assert gridDist >0, "The grid distance should be positive."
            ngrid = ngrid - 1

            skdict["Distance"][isktype] = torch.arange(1,ngrid+1)*gridDist * 0.529177249

            HSvals = torch.zeros([ngrid, NumHvals * 2])
            atomtypes = isktype.split(sep='-')
            if atomtypes[0]==atomtypes[1]:
                log.info('This file is a Homo-nuclear case!')
                # Line 2 for Homo-nuclear case
                datline = format_readline(data[1])
                # Ed Ep Es, spe, Ud Up Us, Od Op Os.
                # order from d p s -> s p d.
                OnSiteEs = torch.tensor([float(datline[2 - ish]) for ish in range(MaxShells)])
                # HubdU[atomtypes[0]] = np.array([float(datline[6 - ish]) for ish in range(MaxShells)])
                # Occu[atomtypes[0]]  = np.array([float(datline[9 - ish]) for ish in range(MaxShells)])

                skdict["OnsiteE"][atomtypes[0]] = OnSiteEs *  13.605662285137 * 2

                for il in range(3, 3 + ngrid):
                    datline = format_readline(data[il])
                    HSvals[il - 3] = torch.tensor([float(val) for val in datline[0:2 * NumHvals]])
                
            else:
                log.info('This is for Hetero-nuclear case!')
                for il in range(2, 2 + ngrid):
                    datline = format_readline(data[il])
                    HSvals[il - 2] = torch.tensor([float(val) for val in datline[0:2 * NumHvals]])

            # HSintgrl[isktype] = HSvals
            skdict['Hopping'][isktype] = HSvals[:,:NumHvals].T * 13.605662285137 * 2
            skdict['Overlap'][isktype] = HSvals[:,NumHvals:].T
            

        return skdict
    
    def format_skparams(self, skdict):
        """
            Formats the given skdict to ensure that all parameters have the same length and values for x. 
            And pick out the orbital-pair sk integrals according to the idp_sk.orbpairtype_maps and saveed in the same order.

            Args:
                skdict (dict): A dictionary containing the parameters for the sk fils.

            Returns:
                dict: A formatted skdict.
                {"Distance": torch.tensor,
                    "OnsiteE": dict,
                    "Hopping": dict,
                    "Overlap": dict
                }
                "OnsiteE": {atomtype: torch.tensor[Es,Ep,Ed]},
                "Hopping": {bondtype: torch.tensor[orbpair, x]},
                "Overlap": {bondtype: torch.tensor[orbpair, x]},
        """
        # 检查所有的参数格式是否正确:
        if isinstance(skdict["Distance"],torch.Tensor):
            assert isinstance(skdict['Hopping'], torch.Tensor)
            assert isinstance(skdict['Overlap'], torch.Tensor)
            assert isinstance(skdict['OnsiteE'], torch.Tensor)

            assert len(skdict["Distance"].shape) == 1

            assert len(skdict['Hopping'].shape) == len(skdict['Overlap'].shape) == len(skdict['OnsiteE'].shape)== 3
            
            assert skdict['Hopping'].shape[0] == skdict['Overlap'].shape[0] == len(self.idp_sk.bond_types)
            assert skdict['Hopping'].shape[1] == skdict['Overlap'].shape[1] == self.idp_sk.reduced_matrix_element
            assert skdict['Hopping'].shape[2] == skdict['Overlap'].shape[2] == len(skdict["Distance"])
            
            assert skdict['OnsiteE'].shape[0] == self.idp_sk.num_types
            assert skdict['OnsiteE'].shape[1] == self.idp_sk.n_onsite_Es
            assert skdict['OnsiteE'].shape[2] == 1

            return skdict
        else:
            assert isinstance(skdict["Distance"], dict), "The Distance should be a dict or a torch.tensor."

        # 固定 x 的长度。使得所有的参数都是具有相同的x的长度和数值。
        x_min = []
        x_max = []
        x_num = []
        for ibtype in skdict["Distance"].keys():
            x_min.append(skdict["Distance"][ibtype].min().item())
            x_max.append(skdict["Distance"][ibtype].max().item())
            assert len(skdict["Distance"][ibtype].shape) == 1
            x_num.append(len(skdict["Distance"][ibtype]))

        x_min = torch.tensor(x_min).max()
        x_max = torch.tensor(x_max).min()
        x_num = torch.tensor(x_num).min()
        xlist_all = torch.linspace(x_min, x_max, x_num)

        format_skdict = {}
        format_skdict['Distance']= xlist_all
        # format_skdict['OnsiteE'].update(skdict['OnsiteE'])

        assert skdict['Hopping'].keys() == skdict['Overlap'].keys()

        onsiteE_params = torch.zeros([self.idp_sk.num_types, self.idp_sk.n_onsite_Es])
        for asym, idx in self.idp_sk.chemical_symbol_to_type.items():
            for ot in self.idp_sk.basis[asym]:
                fot = self.idp_sk.basis_to_full_basis[asym][ot]
                indt = self.onsite_orb_map[fot]
                onsiteE_params[idx][self.idp_sk.skonsite_maps[fot]] = skdict['OnsiteE'][asym][indt]   
        onsiteE_params = onsiteE_params.reshape([self.idp_sk.num_types, self.idp_sk.n_onsite_Es, 1])
        
        hopping_params = torch.zeros([len(self.idp_sk.bond_types), self.idp_sk.reduced_matrix_element, len(xlist_all)])
        overlap_params = torch.zeros([len(self.idp_sk.bond_types), self.idp_sk.reduced_matrix_element, len(xlist_all)])

        for ibtype in self.idp_sk.bond_types:
            idx = self.idp_sk.bond_to_type[ibtype]
            xx = skdict["Distance"][ibtype]
            hh = skdict['Hopping'][ibtype]
            ss = skdict['Overlap'][ibtype]
            assert hh.shape[0] == ss.shape[0] == 10
            
            xx = xx.reshape(1, -1).repeat(10, 1)
            xx_int = xlist_all.reshape([1,-1]).repeat(10, 1)

            intp = Interp1D(x=xx, method='linear')
            hh_intp = intp(xq=xx_int, y=hh)
            ss_intp = intp(xq=xx_int, y=ss)
            
            for ipt in self.idp_sk.orbpairtype_maps.keys():
                slc = self.idp_sk.orbpairtype_maps[ipt]
                hopping_params[idx][slc] = hh_intp[self.intgl_order[ipt]]
                overlap_params[idx][slc] = ss_intp[self.intgl_order[ipt]]



        format_skdict['Hopping'] = hopping_params
        format_skdict['Overlap'] = overlap_params
        format_skdict['OnsiteE'] = onsiteE_params

        return format_skdict

