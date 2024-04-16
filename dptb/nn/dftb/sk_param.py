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
    def __init__(self,
                basis: Dict[str, Union[str, list]]=None,
                idp_sk: Union[OrbitalMapper, None]=None,
                sk_path: str=None) -> None:



        if basis is not None:
            self.idp_sk = OrbitalMapper(basis, method="sktb")
            if idp_sk is not None:
                assert idp_sk.basis == self.idp_sk.basis, "The basis of idp and basis should be the same."
        else:
            assert idp_sk is not None, "Either basis or idp should be provided."
            self.idp_sk = idp_sk

        assert len(self.idp_sk.full_basis)<=3, "The dftb mode only supports 1s, 1p, 1d orbitals. maxmum of orbitals is 3." 
        for iorb in self.idp_sk.full_basis:
            assert iorb in ['1s','1p','1d'], "The dftb mode only supports 1s, 1p, 1d orbitals."

        bond_types = self.idp_sk.bond_types

        assert sk_path is not None, "You need to provide the sk_path."
                
        if '.' in sk_path and sk_path.split('.')[-1] == 'pth':
            log.info('Loading the skdict from the sk_path pth file......')
            skdata = torch.load(sk_path)
            for ibtype in bond_types:
                if ibtype not in skdata:
                    log.error("The bond type: " + ibtype + " is not in the skdict.")
                    sys.exit()
        else:
            log.info('Reading the skfiles from the sk_path......')
            skfiles = {}
            for ibtype in bond_types:
                if not os.path.exists(sk_path + '/' + ibtype + '.skf'):
                    log.error('Didn\'t find the skfile: ' + sk_path + '/' + ibtype + '.skf')
                    sys.exit()
                else:
                    skfiles[ibtype] = sk_path + '/' + ibtype + '.skf'

            skdata = self.read_skfiles(skfiles)
        
        assert isinstance(skdata['Distance'], dict), "The initial skdata should be raw data, directly loaded from the skfiles."

        self.skdict = self.format_skparams(skdata)

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

            skdict["Distance"][isktype] = torch.reshape(torch.arange(1,ngrid+1)*gridDist * 0.529177249, [1,-1])

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
        # 固定 x 的长度。使得所有的参数都是具有相同的x的长度和数值。
        x_min = []
        x_max = []
        x_num = []
        for ibtype in skdict["Distance"].keys():
            x_min.append(skdict["Distance"][ibtype].min().item())
            x_max.append(skdict["Distance"][ibtype].max().item())
            x_num.append(skdict["Distance"][ibtype].shape[1])

        x_min = torch.tensor(x_min).max()
        x_max = torch.tensor(x_max).min()
        x_num = torch.tensor(x_num).min()
        xlist_all = torch.linspace(x_min, x_max, x_num).reshape([1,-1])

        format_skdict = {}
        format_skdict['OnsiteE'] = {}
        format_skdict['Hopping'] = {}
        format_skdict['Overlap'] = {}
        format_skdict['Distance']= xlist_all

        format_skdict['OnsiteE'].update(skdict['OnsiteE'])

        assert skdict['Hopping'].keys() == skdict['Overlap'].keys()

        for ibtype in skdict['Hopping'].keys(): # ibtype: 'C-C'
            xx = skdict["Distance"][ibtype]
            hh = skdict['Hopping'][ibtype]
            ss = skdict['Overlap'][ibtype]
            assert hh.shape[0] == ss.shape[0] == 10
            
            # 提取需要的轨道积分:
            assert self.idp_sk.reduced_matrix_element <= 10, "The reduced_matrix_element should be no more than 10."
            hh_pick = torch.zeros([self.idp_sk.reduced_matrix_element, hh.shape[1]])
            ss_pick = torch.zeros([self.idp_sk.reduced_matrix_element, ss.shape[1]])
            for ipt in self.idp_sk.orbpairtype_maps.keys():
                slc = self.idp_sk.orbpairtype_maps[ipt]
                hh_pick[slc] = hh[self.intgl_order[ipt]]
                ss_pick[slc] = ss[self.intgl_order[ipt]]

            num_intgrls = hh_pick.shape[0] 
            xx = torch.tile(xx.reshape([1,-1]), (num_intgrls,1))
            xx_int = torch.tile(xlist_all, (num_intgrls,1))
            intp = Interp1D(x=xx, method='linear')

            format_skdict['Hopping'][ibtype] = intp(xq=xx_int, y=hh_pick)
            format_skdict['Overlap'][ibtype] = intp(xq=xx_int, y=ss_pick)

        return format_skdict

