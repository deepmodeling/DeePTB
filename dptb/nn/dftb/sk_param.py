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
    def __init__(self,
                basis: Dict[str, Union[str, list]]=None,
                idp_sk: Union[OrbitalMapper, None]=None,
                sk_path: str=None,
                skdict:  str=None) -> None:
    
        if basis is not None:
            self.idp_sk = OrbitalMapper(basis, method="sktb")
            if idp_sk is not None:
                assert idp_sk.basis == self.idp_sk.basis, "The basis of idp and basis should be the same."
        else:
            assert idp_sk is not None, "Either basis or idp should be provided."
            self.idp_sk = idp_sk

        bond_types = self.idp_sk.bond_types
                
        if skdict is not None:
            assert skdict.split(".")[-1] == "pth", "The skdict should be a .pth file."
            skdata = torch.load(skdict)
            self.skdict = {} 
            for ibtype in bond_types:
                if ibtype not in skdata:
                    log.error("The bond type: " + ibtype + " is not in the skdict.")
                    sys.exit()

        elif sk_path is not None:
            skfiles = {}
            for ibtype in bond_types:
                if not os.path.exists(sk_path + '/' + ibtype + '.skf'):
                    log.error('Didn\'t find the skfile: ' + sk_path + '/' + ibtype + '.skf')
                    sys.exit()
                else:
                    skfiles[ibtype] = sk_path + '/' + ibtype + '.skf'

            skdata = self.read_skfiles(skfiles)
            
        else:
            log.error("You need to provide either the sk_path or the skdict.")
            sys.exit()

        # check the skdata is in the correct format

        #self.skdict = self.format_skparams(skdata)
        self.skdict = skdata

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
        for isktype in skfiletypes:
            skdict[isktype] = {}
            filename = skfiles[isktype]
            log.info('Reading SlaterKoster File......')
            log.info(' ' + filename)
            fr = open(filename)
            data = fr.readlines()
            fr.close()
            # Line 1
            datline = format_readline(data[0])
            gridDist, ngrid = float(datline[0]), int(datline[1])
            ngrid = ngrid - 1

            skdict[isktype]['Distlist'] = torch.reshape(torch.arange(1,ngrid+1)*gridDist * 0.529177249, [1,-1])

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

                skdict[isktype]['OnSiteE'] = OnSiteEs *  13.605662285137 * 2
                
                for il in range(3, 3 + ngrid):
                    datline = format_readline(data[il])
                    HSvals[il - 3] = torch.tensor([float(val) for val in datline[0:2 * NumHvals]])
                
            else:
                log.info('This is for Hetero-nuclear case!')
                skdict[isktype]['OnSiteE'] = None
                for il in range(2, 2 + ngrid):
                    datline = format_readline(data[il])
                    HSvals[il - 2] = torch.tensor([float(val) for val in datline[0:2 * NumHvals]])

            # HSintgrl[isktype] = HSvals
            skdict[isktype]['Hintgrl'] = torch.as_tensor(HSvals[:,:NumHvals]).T * 13.605662285137 * 2
            skdict[isktype]['Sintgrl'] = torch.as_tensor(HSvals[:,NumHvals:]).T
            

        return skdict
    
    @classmethod
    def format_skparams(self, skdict):
        '''It formats the skdict to the format that is used in the DeePTB model.

        Parameters
        ----------
        skdict
            the skdict from the read_skfiles method.

        Returns
        -------
        skparams
            the formatted skparams.
        '''
        # 固定 x 的长度。使得所有的参数都是具有相同的x的长度和数值。
        x_min = []
        x_max = []
        x_num = []
        for ibtype in skdict:
            x_min.append(skdict[ibtype]['Distlist'].min().item())
            x_max.append(skdict[ibtype]['Distlist'].max().item())
            x_num.append(skdict[ibtype]['Distlist'].shape[1])

        x_min = torch.tensor(x_min).max()
        x_max = torch.tensor(x_max).min()
        x_num = torch.tensor(x_num).min()
        xlist_all = torch.linspace(x_min, x_max, x_num)

        format_skdict = {}
        for ibtype in skdict:
            format_skdict[ibtype] = {}
            format_skdict[ibtype]['Distlist'] = xlist_all

            xx = skdict[ibtype]['Distlist']
            hh = skdict[ibtype]['Hintgrl']
            ss = skdict[ibtype]['Hintgrl']
            assert hh.shape[0] == ss.shape[0] == 10
            xx = torch.tile(xx.reshape([1,-1]), (10,1))
            xx_int = torch.tile(xlist_all.reshape([1,-1]), (10,1))
            intp = Interp1D(x=xx, method='linear')

            format_skdict[ibtype]['Hintgrl'] = intp(xq=xx_int, y=hh)
            format_skdict[ibtype]['Sintgrl'] = intp(xq=xx_int, y=ss)
            format_skdict[ibtype]['OnSiteE'] = skdict[ibtype]['OnSiteE']

        return format_skdict

