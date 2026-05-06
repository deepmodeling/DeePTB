import os
import sys
import torch
from dptb.utils.tools import format_readline
from dptb.utils.constants import NumHvals,MaxShells,Harte2eV,Bohr2Ang
import logging
from typing import Tuple, Union, Dict
from dptb.data.transforms import OrbitalMapper
from dptb.utils._xitorch.interpolate import Interp1D
import numpy as np
from dptb.nn.sktb.cov_radiiDB import Covalent_radii
from dptb.nn.dftb.interp import Repcurve

log = logging.getLogger(__name__)

class SKParam:
    """
    Container and formatter for Slater-Koster (SK) parameters used in DFTB.

    This class loads SK data from .skf files or a packed .pth dict, unifies all per-bond
    distance grids into a single common grid, reorders integrals according to the
    OrbitalMapper's reduced ordering, and exposes a standardized tensor dictionary
    for model usage. It can also estimate recommended per-bond r_min/r_max cutoffs and
    write formatted parameters back to .skf files.

    Parameters
    ----------
    basis : dict[str, str | list] | None
        Mapping from chemical symbol to orbital specification.
        If provided, an OrbitalMapper will be constructed (method="sktb").
        Provide either basis or idp_sk. If both are provided, their bases must match.
    idp_sk : OrbitalMapper | None
        A pre-built OrbitalMapper. Must be consistent with basis if both are provided.
    skdata : str | dict
        - str:
          - Path ending with ".pth": load a packed dict containing 'Distance', 'Hopping',
            'Overlap', 'OnsiteE' (and optionally 'HubdU', 'Occu', 'Mass').
          - Directory path: read <A-B>.skf files for all bond types in the basis.
        - dict: a raw skdict (as returned by read_skfiles) with per-bond grids.
    cal_rcuts : bool, default False
        If True, compute per-bond recommended r_min and r_max and store in bond_r_min/bond_r_max.
    device : str | torch.device, default 'cpu'
        Preferred device for tensors produced/managed by this instance.
    dtype : torch.dtype | str, default torch.float32
        Preferred floating precision (e.g., torch.float32).

    Attributes
    ----------
    intgl_order : dict[str, slice]
        Mapping from integral group to original 10 SK positions:
        's-s','s-p','s-d','p-p','p-d','d-d'.
    support_full_basis : list[str]
        Supported full shells: ['1s', '1p', '1d'].
    onsite_orb_map : dict[str, int]
        Index mapping for onsite energy triplet per full shell: {'1s': 0, '1p': 1, '1d': 2}.
    idp_sk : OrbitalMapper
        The OrbitalMapper built from basis or provided by user. Its orb-pair and onsite maps
        are used to pack/unpack reduced integral blocks.
    skdict : dict
        Standardized parameter dict after formatting:
        - 'Distance': torch.Tensor[n_x] (Å)
        - 'Hopping' : torch.Tensor[n_bond, reduced, n_x] (eV)
        - 'Overlap' : torch.Tensor[n_bond, reduced, n_x] (dimensionless)
        - 'OnsiteE' : torch.Tensor[n_types, n_onsite, 1] (eV)
        - 'HubdU'   : torch.Tensor[n_types, n_onsite, 1] (eV)
        - 'Occu'    : torch.Tensor[n_types, n_onsite, 1] (dimensionless)
        - 'Mass'    : torch.Tensor[n_types, 1] (atomic mass unit)
        - 'Highest_Occu_U': torch.Tensor[n_types, 1, 1] (for each element, the Hubbard U
          corresponding to the highest-energy orbital among those with Occu > 0; 0 if none)
    bond_r_min, bond_r_max : dict[str, float] | None
        If cal_rcuts=True, dictionaries of recommended min/max bond distances (Å) per bond type.
    device, dtype : as provided.

    Notes
    -----
    - Units: distances are converted Bohr→angstrom; energies are converted Hartree→eV.
    - Grid unification uses linear interpolation; out-of-range values are set to 0.
    - Reduced integral ordering is given by idp_sk.orbpairtype_maps and mapped to the original
      10 SK integrals via intgl_order.
    - read_skfiles expects <A-B>.skf for all bond types from the basis.

    Raises
    ------
    AssertionError
        Inconsistent inputs or unsupported basis (only 1s/1p/1d shells; max 3 shells).
    FileNotFoundError
        A required .skf file was not found.
    ValueError
        skdata is invalid, missing required keys, or contains inconsistent shapes.
    """
    
    # Mapping from integral type to slice in the 10 Slater-Koster integrals
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
                skdata: Union[str,dict] = None,
                cal_rcuts: bool = False,
                cal_atomic_rmin_rmax: bool = False,
                device='cpu',
                dtype=torch.float32) -> None:
        
        self.device = device
        if isinstance(dtype, str):
            dtype =  getattr(torch, dtype)
        self.dtype = dtype
        
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
                skdict = torch.load(skdata, weights_only=False)
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
        
        if cal_rcuts:
            self.bond_r_min, self.bond_r_max = cal_rmin_rmax_bondwise(skdata=skdict)
        else:
            self.bond_r_min = None
            self.bond_r_max = None

        if cal_atomic_rmin_rmax:
            self.atomic_r_min, self.atomic_r_max = cal_rmin_rmax(skdata=skdict)
        else:
            self.atomic_r_min = None
            self.atomic_r_max = None    
        
        self.skdict = self.format_skparams(skdict)

    @classmethod
    def read_skfiles(cls, skfiles):
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
        skdict: dict with the following top-level keys and shapes
            - 'Distance': { bondtype: torch.Tensor[ngrid] }
                Unified per-bond distance grid in angstrom (from Bohr via * Bohr2Ang)

            - 'Hopping' : { bondtype: torch.Tensor[ngrid, NumHvals] }
                Slater-Koster hopping integrals in eV (from Hartree via * Harte2eV)
                Note that the ordering of NumHvals is as follows: Hdd0 Hdd1 Hdd2 Hpd0 Hpd1 Hpp0 Hpp1 Hsp0 Hsp1 Hss0

            - 'Overlap' : { bondtype: torch.Tensor[ngrid, NumHvals] }
                Slater-Koster overlap integrals (unitless)
                Note that the ordering of NumHvals is as follows: Sdd0 Sdd1 Sdd2 Spd0 Spd1 Spp0 Spp1 Ssp0 Ssp1 Sss0

            - 'OnsiteE' : { symbol: torch.Tensor[Es, Ep, Ed] }
                On-site energies for s/p/d shells in eV (from Hartree via * Harte2eV)

            - 'HubdU'   : { symbol: torch.Tensor[Us, Up, Ud] }
                Hubbard U for s/p/d shells in eV  (from Hartree via * Harte2eV)

            - 'Occu'    : { symbol: torch.Tensor[Os, Op, Od] }
                Ground-state occupations for s/p/d shells (dimensionless)

            - 'Mass'    : { symbol: torch.Tensor }
                Atomic mass for the given atom type (in atomic mass unit)
        '''
        assert isinstance(skfiles, dict)
        skfiletypes = list(skfiles.keys())

        skdict = {}
        skdict['Mass'] = {}
        skdict['OnsiteE'] = {}
        skdict['Hopping']  = {}
        skdict['Overlap']  = {}
        skdict["Distance"] = {}
        skdict["HubdU"] = {}
        skdict["Occu"] = {}
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

            skdict["Distance"][isktype] = torch.arange(1,ngrid+1)*gridDist * Bohr2Ang

            HSvals = torch.zeros([ngrid, NumHvals * 2])
            atomtypes = isktype.split(sep='-')
            if atomtypes[0]==atomtypes[1]:
                log.info('This file is a Homo-nuclear case!')
                # Line 2 for Homo-nuclear case
                datline = format_readline(data[1])
                # Ed Ep Es, spe, Ud Up Us, Od Op Os. 
                # spe is spin polarization error, which is ignored.
                # order from d p s -> s p d.
                OnSiteEs = torch.tensor([float(datline[2 - ish]) for ish in range(MaxShells)]) # Ed Ep Es are onsite energies for d,p and s for the given atom.
                HubdU = torch.tensor([float(datline[6 - ish]) for ish in range(MaxShells)]) # Ud Up Us are Hubbard U values for d,p and s for the given atom.
                Occu  = torch.tensor([float(datline[9 - ish]) for ish in range(MaxShells)]) # Od Op Os are occupations for the neutral atom in the ground state.

                skdict["OnsiteE"][atomtypes[0]] = OnSiteEs *  Harte2eV # from Hartree to eV
                skdict["HubdU"][atomtypes[0]] = HubdU *  Harte2eV # from Hartree to eV
                skdict["Occu"][atomtypes[0]] = Occu

                # Line 3 for Homo-nuclear case: Mass and spline info
                datline = format_readline(data[2])
                skdict["Mass"][atomtypes[0]] = torch.tensor(float(datline[0])) # in atomic mass unit
                
                for il in range(3, 3 + ngrid):
                    datline = format_readline(data[il])
                    HSvals[il - 3] = torch.tensor([float(val) for val in datline[0:2 * NumHvals]])
                
            else:
                log.info('This is for Hetero-nuclear case!')
                for il in range(2, 2 + ngrid):
                    datline = format_readline(data[il])
                    HSvals[il - 2] = torch.tensor([float(val) for val in datline[0:2 * NumHvals]])

            # HSintgrl[isktype] = HSvals
            skdict['Hopping'][isktype] = HSvals[:,:NumHvals].T * Harte2eV  # from Hartree to eV
            skdict['Overlap'][isktype] = HSvals[:,NumHvals:].T
            

        return skdict
    
    def format_skparams(self, skdict):
        """
            Formats the given skdict to ensure that all parameters have the same length and values for x. 
            And pick out the orbital-pair sk integrals according to the idp_sk.orbpairtype_maps and saveed in the same order.

            Parameters:
            ------------
                skdict (dict): A dictionary containing the parameters for the sk fils.

            Returns:
            -------- 
                dict: formatted_skdict with normalized grids and shapes:
                {
                    'Distance': torch.Tensor[n_x],
                        # Unified distance grid in angstrom (1D tensor)

                    'Hopping': torch.Tensor[n_bond_types, reduced_matrix_element, n_x],
                        # Slater-Koster hopping integrals per bond type
                        # reduced_matrix_element is the number of integrals under symmetry reductions
                        # in the order like ['s-s', 's-p', 'p-p'](see idp_sk.orbpairtype_maps)

                    'Overlap': torch.Tensor[n_bond_types, reduced_matrix_element, n_x],
                        # Overlap integrals with the same ordering and grid as 'Hopping'
                        # reduced_matrix_element is the number of integrals under symmetry reductions
                        # in the order like ['s-s', 's-p', 'p-p'](see idp_sk.orbpairtype_maps)

                    'OnsiteE': torch.Tensor[num_types, n_onsite_Es, 1],
                        # On-site energies (eV).

                    'HubdU': torch.Tensor[num_types, n_onsite_Es, 1],
                        # Hubbard U values (eV)

                    'Occu': torch.Tensor[num_types, n_onsite_Es, 1],
                        # Ground-state occupations (dimensionless)
                    
                    'Mass': torch.Tensor[num_types, 1],
                        # Atomic mass for the given atom type (in atomic mass unit)

                    'Highest_Occu_U': torch.Tensor[num_types, 1, 1],
                        # For each element, the U corresponding to the highest-energy orbital among those with Occu > 0
                }

                where n_x is the length of the unified distance grid.
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

        x_min = torch.tensor(x_min).min()
        x_max = torch.tensor(x_max).max()
        x_num = torch.tensor(x_num).max()
        xlist_all = torch.linspace(x_min, x_max, x_num, dtype=self.dtype, device=self.device) # the public x grid for all sk params

        format_skdict = {}
        format_skdict['Distance']= xlist_all
        # format_skdict['OnsiteE'].update(skdict['OnsiteE'])

        assert skdict['Hopping'].keys() == skdict['Overlap'].keys()

        onsiteE_params = torch.zeros([self.idp_sk.num_types, self.idp_sk.n_onsite_Es], dtype=self.dtype, device=self.device)
        mass_params = torch.zeros([self.idp_sk.num_types, 1], dtype=self.dtype, device=self.device)

        if 'Mass' not in skdict:
            log.warning("The Mass parameter is not provided in the skdict. It will be set to zero.")
        else:
            for asym, idx in self.idp_sk.chemical_symbol_to_type.items():
                mass_params[idx][0] = skdict['Mass'][asym]

        # hubbard values and Occu_values have the same length of onsite e
        Hubbard_values = torch.zeros([self.idp_sk.num_types, self.idp_sk.n_onsite_Es], dtype=self.dtype, device=self.device)
        Occu_values = torch.zeros([self.idp_sk.num_types, self.idp_sk.n_onsite_Es], dtype=self.dtype, device=self.device)
        for asym, idx in self.idp_sk.chemical_symbol_to_type.items():
            for ot in self.idp_sk.basis[asym]:
                fot = self.idp_sk.basis_to_full_basis[asym][ot]
                indt = self.onsite_orb_map[fot]                    
                onsiteE_params[idx][self.idp_sk.skonsite_maps[fot+"-"+fot]] = skdict['OnsiteE'][asym][indt]   
                Hubbard_values[idx][self.idp_sk.skonsite_maps[fot+"-"+fot]] = skdict['HubdU'][asym][indt]
                Occu_values[idx][self.idp_sk.skonsite_maps[fot+"-"+fot]] = skdict['Occu'][asym][indt]
        onsiteE_params = onsiteE_params.reshape([self.idp_sk.num_types, self.idp_sk.n_onsite_Es, 1])
        Hubbard_values = Hubbard_values.reshape([self.idp_sk.num_types, self.idp_sk.n_onsite_Es, 1])
        Occu_values = Occu_values.reshape([self.idp_sk.num_types, self.idp_sk.n_onsite_Es, 1])

        hopping_params = torch.zeros([len(self.idp_sk.bond_types), self.idp_sk.reduced_matrix_element, len(xlist_all)], dtype=self.dtype, device=self.device)
        overlap_params = torch.zeros([len(self.idp_sk.bond_types), self.idp_sk.reduced_matrix_element, len(xlist_all)], dtype=self.dtype, device=self.device)

        for ibtype in self.idp_sk.bond_types:
            idx = self.idp_sk.bond_to_type[ibtype]
            xx = skdict["Distance"][ibtype]
            hh = skdict['Hopping'][ibtype]
            ss = skdict['Overlap'][ibtype]
            assert hh.shape[0] == ss.shape[0] == 10
            
            xx = xx.reshape(1, -1).repeat(10, 1)
            xx_int = xlist_all.reshape([1,-1]).repeat(10, 1) # the public x grid for interpolation
            intp = Interp1D(x=xx, method='linear')

            x_min = xx.min().item()
            x_max = xx.max().item()
            mask_in_range = (xx_int >= x_min) & (xx_int <= x_max)
            mask_out_range = ~mask_in_range
            if mask_out_range.any():
                xx_tmp = xx_int.clone()
                xx_tmp[mask_out_range] = (x_min + x_max) / 2
                hh_intp = intp(xq=xx_tmp, y=hh)
                ss_intp = intp(xq=xx_tmp, y=ss)
                hh_intp[mask_out_range] = 0.0
                ss_intp[mask_out_range] = 0.0 # assign zero to the out-of-range points
            else:
                hh_intp = intp(xq=xx_int, y=hh)
                ss_intp = intp(xq=xx_int, y=ss)
            
            for ipt in self.idp_sk.orbpairtype_maps.keys():
                slc = self.idp_sk.orbpairtype_maps[ipt]
                hopping_params[idx][slc] = hh_intp[self.intgl_order[ipt]]
                overlap_params[idx][slc] = ss_intp[self.intgl_order[ipt]]

        format_skdict['Hopping'] = hopping_params
        format_skdict['Overlap'] = overlap_params
        format_skdict['OnsiteE'] = onsiteE_params
        format_skdict['Mass'] = mass_params # If Mass is not provided, it will be zero.
        format_skdict['HubdU'] = Hubbard_values
        format_skdict['Occu'] = Occu_values

        # get highest occpy Hubbard U
        # 1. 创建一个 Occu 掩码，用于筛选出每个 ia 中 Occupation 大于 0 的位置  
        mask = format_skdict['Occu'] > 0
        onsite_e_for_argmax = format_skdict['OnsiteE'].clone()
        # 3. 将不满足条件的位置设置为负无穷大  torch.where(condition, value_if_true, value_if_false) 这样在 argmax 中，这些被屏蔽的位置就永远不会是最大值
        onsite_e_for_argmax = torch.where(mask, onsite_e_for_argmax, torch.tensor(float('-inf'), dtype=self.dtype, device=self.device))
        # 4. 沿每个 ia (即每行) 查找最大值的索引 dim=1 表示在第二个维度 (轨道维度) 上操作 =True 保持输出维度为 (num_ia, 1)，方便后续 gather 操作
        max_indices = torch.argmax(onsite_e_for_argmax, dim=1, keepdim=True)
        # 5. 使用 gather 从 HubdU 中并行取出所有对应的值 torch.gather(input, dim, index) -> 根据 index 从 input 中取值
        highest_occpy_hubbard_u = torch.gather(format_skdict['HubdU'], 1, max_indices)
        # 6. 处理没有任何元素满足条件的行 (这些行的最大值来自 -inf)  找到这些行，并将它们的 Hubbard U 值设为 0 mask.any(dim=1) 检查每一行是否至少有一个 True
        valid_rows_mask = mask.any(dim=1, keepdim=True)
        highest_occpy_hubbard_u = highest_occpy_hubbard_u * valid_rows_mask.to(dtype=self.dtype)
        
        # highest_occpy_hubbard_u shape [n_atom_symbol, 1, 1]
        format_skdict['Highest_Occu_U'] = highest_occpy_hubbard_u
        
        return format_skdict

    @classmethod
    def formatted_to_skdict(cls, formatted_skdict: dict, idp_sk: OrbitalMapper = None) -> dict:
        """
        Convert a formatted skdict (unified grid) back to the raw skdict structure
        used by read_skfiles: per-bond distances and NumHvals Slater-Koster integrals ordering.

        Parameters:
        ------------
        - formatted_skdict: dict with keys 'Distance' (1D tensor), 'Hopping', 'Overlap',
                            'OnsiteE', 'HubdU', 'Occu'. Shapes follow format_skparams outputs.
        - idp_sk: OrbitalMapper instance used for mapping bond types and orbital pairs.
        Returns:
        ------------
        - skdict: dict compatible with read_skfiles outputs:
            {
              'Distance': { bondtype: 1D tensor },
                'Hopping' : { bondtype: tensor[ngrid, NumHvals] in original NumHvals-int order },
                'Overlap' : { bondtype: tensor[ngrid, NumHvals] in original NumHvals-int order },
              'OnsiteE' : { symbol: tensor[Es, Ep, Ed] },
              'HubdU'   : { symbol: tensor[Us, Up, Ud] },
              'Occu'    : { symbol: tensor[Os, Op, Od] },
            }
        """

        if idp_sk is None:
            raise ValueError("idp_sk must be provided for formatted_to_skdict.")
        assert isinstance(idp_sk, OrbitalMapper), "idp_sk must be an instance of OrbitalMapper"
        log.info("The provided idp_sk is used for mapping bond types and orbital pairs.")

        assert isinstance(formatted_skdict, dict), "formatted_skdict must be a dict"
        assert isinstance(formatted_skdict['Hopping'], torch.Tensor)
        assert isinstance(formatted_skdict['Overlap'], torch.Tensor)
        assert isinstance(formatted_skdict['OnsiteE'], torch.Tensor)
        assert isinstance(formatted_skdict['Mass'], torch.Tensor)

        assert len(formatted_skdict["Distance"].shape) == 1

        assert (
            len(formatted_skdict['Hopping'].shape)
            == len(formatted_skdict['Overlap'].shape)
            == len(formatted_skdict['OnsiteE'].shape)
            == 3
        )

        assert formatted_skdict['Hopping'].shape[0] == formatted_skdict['Overlap'].shape[0] == len(idp_sk.bond_types)
        assert formatted_skdict['Hopping'].shape[1] == formatted_skdict['Overlap'].shape[1] == idp_sk.reduced_matrix_element
        assert formatted_skdict['Hopping'].shape[2] == formatted_skdict['Overlap'].shape[2] == len(formatted_skdict["Distance"])

        assert formatted_skdict['OnsiteE'].shape[0] == idp_sk.num_types
        assert formatted_skdict['OnsiteE'].shape[1] == idp_sk.n_onsite_Es
        assert formatted_skdict['OnsiteE'].shape[2] == 1

        xlist_all = formatted_skdict['Distance']  # 1D tensor
        hopping_fmt = formatted_skdict['Hopping']  # [n_bond_types, reduced, len(x)]
        overlap_fmt = formatted_skdict['Overlap']  # same shape

        # Prepare outputs
        skdict = {
            'Distance': {},
            'Hopping': {},
            'Overlap': {},
            'OnsiteE': {},
            'HubdU': {},
            'Occu': {},
            'Mass': {},
        }


        # Rebuild original integral order in skdict
        for btype in idp_sk.bond_types:
            # Per-bond Distance: assign the unified grid to all bond types
            skdict['Distance'][btype] = xlist_all.clone()
            ib = idp_sk.bond_to_type[btype]
            ngrid = len(xlist_all)
            H_vals = torch.zeros(NumHvals, ngrid, dtype=hopping_fmt.dtype, device=hopping_fmt.device)
            S_vals = torch.zeros(NumHvals, ngrid, dtype=overlap_fmt.dtype, device=overlap_fmt.device)

            for ipt, slc in idp_sk.orbpairtype_maps.items():
                assert ipt in cls.intgl_order, f"Integral type {ipt} not recognized in intgl_order."
                # Assign reduced block back to the original 10-int positions
                H_vals[cls.intgl_order[ipt]] = hopping_fmt[ib][slc]
                S_vals[cls.intgl_order[ipt]] = overlap_fmt[ib][slc]

            skdict['Hopping'][btype] = H_vals
            skdict['Overlap'][btype] = S_vals

        # Onsite-related per-symbol
        # Internal formatted order is [s, p, d] along dim=1
        mass_fmt = formatted_skdict['Mass']  # [n_types, 1]
        onsite_fmt = formatted_skdict['OnsiteE']  # [n_types, n_onsite, 1]
        hubdu_fmt = formatted_skdict.get('HubdU', None)
        occu_fmt = formatted_skdict.get('Occu', None)

        # Build index helpers for s/p/d in the formatted tensor using skonsite_maps
        # Keys in skonsite_maps look like '1s-1s', '1p-1p', '1d-1d'
        def get_idx_if_exists(orb_label: str):
            key = f"{orb_label}-{orb_label}"
            return idp_sk.skonsite_maps[key] if key in idp_sk.skonsite_maps else None

        s_idx = get_idx_if_exists('1s')
        p_idx = get_idx_if_exists('1p')
        d_idx = get_idx_if_exists('1d')

        for asym, t_idx in idp_sk.chemical_symbol_to_type.items():
            # Extract [s, p, d] values when existing; fill 0 if missing
            def extract_triplet(tensor3d):
                vals = []
                for idx in (s_idx, p_idx, d_idx):
                    if idx is None:
                        vals.append(torch.tensor([0.0], dtype=tensor3d.dtype, device=tensor3d.device))
                    else:
                        vals.append(tensor3d[t_idx, idx, 0])
                return torch.stack(vals).flatten()  # [Es, Ep, Ed] in s,p,d order

            skdict['OnsiteE'][asym] = extract_triplet(onsite_fmt)
            skdict['Mass'][asym] = mass_fmt[t_idx, 0]
            if hubdu_fmt is not None:
                skdict['HubdU'][asym] = extract_triplet(hubdu_fmt)
            else:
                skdict['HubdU'][asym] = torch.zeros_like(skdict['OnsiteE'][asym])
            if occu_fmt is not None:
                skdict['Occu'][asym] = extract_triplet(occu_fmt)
            else:
                skdict['Occu'][asym] = torch.zeros_like(skdict['OnsiteE'][asym])

        return skdict
    
    @staticmethod
    def compress_zero_line(values: list, fmt: str) -> str:
        """Format a list of floats with fmt, then compress consecutive zeros into n*0.
        Only runs with length >=2 are compressed; single zero stays as '0'.
        """
        # precomputed zero tokens for compression
        zero_token = fmt.format(0.0)
        neg_zero_token = fmt.format(-0.0)
        formatted = [fmt.format(v) for v in values]
        tokens = []
        run = 0
        for s in formatted:
            if s == zero_token or s == neg_zero_token:
                run += 1
            else:
                if run >= 2:
                    tokens.append(f"{run}*0.0")
                elif run == 1:
                    tokens.append("0.0")
                run = 0
                tokens.append(s)
        if run >= 2:
            tokens.append(f"{run}*0.0")
        elif run == 1:
            tokens.append("0.0")
        return "    ".join(tokens)

    @classmethod
    def write_skf_from_formatted(
        cls,
        formatted_skdict: dict,
        out_dir: str,
        idp_sk: Union[OrbitalMapper, None] = None,
        basis: Union[Dict[str, Union[str, list]], None] = None,
        overwrite: bool = False,
        spe: float = 0.0,
        float_precision: int = 6,
        add_rep: bool = False,
        **kwargs
    ) -> None:
        """
        Write .skf files from a formatted skdict. This function reconstructs the raw skdict
        (per-bond distances and NumHvals-integral order) and writes one file per bond type.

        Parameters:
        ------------
        - formatted_skdict (dict): output of format_skparams with keys:
            - 'Distance': 1D tensor of length ngrid (Å)
            - 'Hopping':  tensor [n_bond_types, reduced_matrix_element, ngrid] (eV)
            - 'Overlap':  tensor [n_bond_types, reduced_matrix_element, ngrid] (unitless)
            - 'OnsiteE':  tensor [num_types, n_onsite_Es, 1] (eV)
            - 'HubdU':    tensor [num_types, n_onsite_Es, 1] (eV)
            - 'Occu':     tensor [num_types, n_onsite_Es, 1] (dimensionless)
        - out_dir (str): directory to place <A-B>.skf files
        - idp_sk / basis: provide either an OrbitalMapper (preferred) or basis to create one
        - overwrite (bool): allow overwriting existing files
        - spe (float): value to write as the 'spe' field in homo files' second line (unused in reader)
        - float_precision (int): number of decimals for floats. Default is 6.
        - add_rep (bool): whether to add a repulsive potential curve
        - kwargs: additional keyword arguments for repulsive potential curve, including
            - 'sigma_rep' (float): empirical parameter representing the size of each atom
            - 'r_min' (float): minimum distance for the repulsive potential, in Angstrom
            - 'r_max' (float): maximum distance for the repulsive potential, in Angstrom
            - 'num_points' (int): number of points to sample for the repulsive potential curve
        """
        assert idp_sk is not None or basis is not None, "Provide idp_sk or basis"
        if idp_sk is None:
            idp_sk = OrbitalMapper(basis, method="sktb")
            idp_sk.get_orbpair_maps()
            idp_sk.get_skonsite_maps()

        os.makedirs(out_dir, exist_ok=True)

        skdict = cls.formatted_to_skdict(formatted_skdict, idp_sk)


        fmt = f"{{:.{float_precision}f}}" # format string for floats

        for btype in idp_sk.bond_types:
            sym_i, sym_j = btype.split('-')
            x = skdict['Distance'][btype]
            Hvals = skdict['Hopping'][btype]
            Svals = skdict['Overlap'][btype]

            # Basic checks
            assert x.ndim == 1
            ngrid = x.numel()
            assert Hvals.shape == (NumHvals, ngrid)
            assert Svals.shape == (NumHvals, ngrid)

            # Header: gridDist in Bohr, ngrid_write = ngrid + 1
            if ngrid > 1:
                diffs = x[1:] - x[:-1]
                delta = diffs.mean().item()
                # warn if the grid is not strictly uniform
                if (diffs - delta).abs().max().item() > 1e-6:
                    log.warning(f"Non-uniform distance grid detected for {btype}; using mean delta for header.")
            else:
                # fallback for single-point grid
                raise ValueError(f"Distance grid for {btype} has only one point; cannot determine grid spacing.")
            grid_dist_bohr = delta / Bohr2Ang
            ngrid_write = ngrid + 1

            out_path = os.path.join(out_dir, f"{btype}.skf")
            if (not overwrite) and os.path.exists(out_path):
                raise FileExistsError(f"File exists: {out_path}. Set overwrite=True to replace.")

            with open(out_path, 'w') as fw:
                # Line 1
                fw.write(f"{fmt.format(grid_dist_bohr)} {ngrid_write}\n")

                # Homo-nuclear: include second line with onsite/hubbard/occu
                if sym_i == sym_j:
                    EsEpEd = skdict['OnsiteE'][sym_i]  # [Es, Ep, Ed] in eV
                    UsUpUd = skdict['HubdU'][sym_i]   # [Us, Up, Ud] in eV
                    OsOpOd = skdict['Occu'][sym_i]    # [Os, Op, Od]

                    # Convert back units and reorder to file order: Ed Ep Es, spe, Ud Up Us, Od Op Os
                    Ed, Ep, Es = (EsEpEd[2] / Harte2eV).item(),\
                                 (EsEpEd[1] / Harte2eV).item(),\
                                 (EsEpEd[0] / Harte2eV).item()
                    Ud, Up, Us = (UsUpUd[2] / Harte2eV).item(),\
                                 (UsUpUd[1] / Harte2eV).item(),\
                                 (UsUpUd[0] / Harte2eV).item()
                    Od, Op, Os = OsOpOd[2].item(), OsOpOd[1].item(), OsOpOd[0].item()

                    line2_vals = [Ed, Ep, Es, spe, Ud, Up, Us, Od, Op, Os]
                    fw.write(" ".join(fmt.format(v) for v in line2_vals) + "\n")

                # Mass line: atomic mass for the given atom type (in atomic mass unit)
                # For now we ignore the repulsion spline coefficients in this line
                mass = skdict['Mass'][sym_i].item()
                fw.write(f"{fmt.format(mass)} 19*0.0 \n")
                
                
                # Integral lines: each has 2*NumHvals numbers: NumHvals H (scaled back), then NumHvals S
                for ig in range(ngrid):
                    h_vals = (Hvals[:, ig] / Harte2eV).tolist()
                    s_vals = Svals[:, ig].tolist()
                    fw.write(cls.compress_zero_line(h_vals + s_vals, fmt) + "\n")

        if add_rep: # add repulsive potential curve to the sk files
            assert 'sigma_rep' in kwargs, \
                "sigma_rep must be provided to add repulsive potential curve."
            sigma_rep = kwargs['sigma_rep']
            r_min = kwargs.get('r_min', 0.25) # unit: Angstrom
            r_max = kwargs.get('r_max', 25.0) # unit: Angstrom
            num_points = kwargs.get('num_points', 800)

            cls.add_rep_curve(
                out_dir=out_dir,
                element_symbols=list(idp_sk.chemical_symbol_to_type.keys()),
                sigma_rep=sigma_rep,
                r_min=r_min,
                r_max=r_max,
                num_points=num_points,
                device=formatted_skdict['Hopping'].device,
                dtype=formatted_skdict['Hopping'].dtype
            )
      
    
    @staticmethod
    def add_rep_curve(
                    out_dir: str,
                    element_symbols: list,
                    sigma_rep: dict,
                    r_min: float = 0.25,
                    r_max: float = 25,
                    num_points: int = 800,
                    device: Union[torch.device, str] = 'cpu',
                    dtype: torch.dtype = torch.float32):
        """
        Add a repulsive potential curve to the Slater-Koster files in the specified directory.

        Parameters:
        ------------
        out_dir (str): The directory where the Slater-Koster files are located.
        element_symbols (list): A list of element symbols for which the repulsive potential is defined.
        sigma_rep (dict): A dictionary mapping element symbols to their corresponding repulsive potential parameters.
        r_min (float): The minimum distance for the repulsive potential curve. Default is 0.25.
        r_max (float): The maximum distance for the repulsive potential curve. Default is 25.
        num_points (int): The number of points in the repulsive potential curve. Default is 800.
        device (torch.device or str): The device to use for tensor computations. Default is 'cpu'.
        dtype (torch.dtype): The data type for tensor computations. Default is torch.float32.
        """
        rep_curve = Repcurve(
                 element_symbols=element_symbols,
                 sigma_rep=sigma_rep,
                 r_min=r_min,
                 r_max=r_max,
                 num_points=num_points,
                 device= device,
                 dtype = dtype)
        
        rep_curve.write_rep_to_skfile(out_dir)

    
def find_first_false(arr):
    """
    Find the index of the first occurrence of False in each row of a 2D array, counting from the end of the row.

    Parameters:
    arr (numpy.ndarray): The input 2D array.

    Returns:
    numpy.ndarray: An array containing the indices of the first occurrence of False in each row.
                   If a row does not contain any False, the corresponding index is set to -1.
    """
    assert arr.ndim == 2
    reversed_arr = np.flip(arr, axis=1)
    reversed_indices = np.argmax(reversed_arr == False, axis=1)
    original_indices = arr.shape[1] - 1 - reversed_indices
    no_false_rows = np.all(reversed_arr, axis=1)
    original_indices[no_false_rows] = -1
    return original_indices


def cal_rmin_rmax_bondwise(skdata):
    """
    Calculate the minimum and maximum bond distances for each pair of atomic symbols.

    This function computes the minimum and maximum bond distances (rmin and rmax) for each 
    pair of atomic symbols based on the provided Slater-Koster data. The minimum bond distance 
    is calculated using the covalent radii of the atoms, and the maximum bond distance is 
    determined from the hopping integrals and distances.

    Parameters:
    skdata (dict): A dictionary containing Slater-Koster data with the following keys:
        - 'OnsiteE': A dictionary where keys are atomic symbols and values are onsite energies.
        - 'Hopping': A dictionary where keys are bond types (e.g., 'C-H') and values are 
                     tensors of hopping integrals.
        - 'Distance': A dictionary where keys are bond types and values are tensors of distances.

    Returns:
    tuple: A tuple containing two dictionaries:
        - atomic_r_min_dict (dict): A dictionary where keys are bond types and values are the 
                                    minimum bond distances.
        - atomic_r_max_dict (dict): A dictionary where keys are bond types and values are the 
                                    maximum bond distances.
    """
    atomic_symbols = list(skdata['OnsiteE'].keys())

    bond_r_max_dict = {}
    bond_r_min_dict = {}
    for isym in atomic_symbols:
        for jsym in atomic_symbols:
            bondtype = isym + '-' + jsym
            inv_bondtype = jsym + '-' + isym

            hopp = skdata['Hopping'][bondtype].numpy()
            dist = skdata['Distance'][bondtype].numpy()
            assert len(dist) == hopp.shape[1]
            # ind = find_first_false(np.abs(hopp)<1e-3*np.abs(hopp).max())
            ind = find_first_false(np.abs(hopp)<1e-2)
            rmax = dist[np.max(ind)].item()
            if inv_bondtype in bond_r_max_dict:
                rmax = max(rmax, bond_r_max_dict[inv_bondtype])
            bond_r_max_dict[bondtype] = round(rmax,2)
            bond_r_max_dict[inv_bondtype] = round(rmax,2)


            bond_r_min_dict[bondtype] = round(0.5 * Covalent_radii[isym] + 0.5 * Covalent_radii[jsym],2)
    return bond_r_min_dict, bond_r_max_dict


def cal_rmin_rmax(skdata):
    """
    Calculate the minimum and maximum atomic radii based on the given skdata.

    Args:
        skdata (dict): Dictionary containing the skdata.

    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary contains the minimum atomic radii
               for each atomic symbol, and the second dictionary contains the maximum atomic radii for each
               atomic symbol.
    """

    atomic_symbols = list(skdata['OnsiteE'].keys())

    # using homo param to determine  the rmax
    atomic_r_max_dict = {}

    for isym in atomic_symbols:
        bondtype = isym + '-' + isym
        hopp = skdata['Hopping'][bondtype].numpy()
        dist = skdata['Distance'][bondtype].numpy()
        assert len(dist) == hopp.shape[1]
        ind = find_first_false(np.abs(hopp)<1e-2)
        rmax = dist[np.max(ind)]
        # rmx 保留两位小数
        atomic_r_max_dict[isym] = round(rmax / 2,2)
    
    # update rmax based on the homo rmax and hetero param.
    for isym in atomic_symbols:
        for jsym in atomic_symbols:
            if isym != jsym:
                bondtype = isym + '-' + jsym
                hopp = skdata['Hopping'][bondtype].numpy()
                dist = skdata['Distance'][bondtype].numpy()
                assert len(dist) == hopp.shape[1]
                ind = find_first_false(np.abs(hopp)<1e-2)
                rmax = dist[np.max(ind)]

                # 按照共价半径比例拆分
                rario = atomic_r_max_dict[isym] / (atomic_r_max_dict[isym] + atomic_r_max_dict[jsym])
                rmax_isym = rmax * rario
                rmax_jsym = rmax * (1-rario)
                if rmax_isym >  atomic_r_max_dict[isym]:
                    atomic_r_max_dict[isym] = round(rmax_isym,2)
                if rmax_jsym >  atomic_r_max_dict[jsym]:
                    atomic_r_max_dict[jsym] = round(rmax_jsym,2)

    atomic_r_min_dict = {}
    for isym in atomic_symbols:
        atomic_r_min_dict[isym] = round(0.5 * Covalent_radii[isym],2)

    return atomic_r_min_dict, atomic_r_max_dict
