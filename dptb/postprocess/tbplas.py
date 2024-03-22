import numpy as np
from dptb.utils.tools import j_must_have, get_neighbours
from dptb.utils.make_kpoints  import ase_kpath, abacus_kpath, vasp_kpath
from ase.io import read
from dptb.utils.constants import atomic_num_dict_r, anglrMId
import ase
import matplotlib.pyplot as plt
import re
import os
import torch
import logging
from dptb.utils.tools import write_skparam
from typing import Optional, Union
from scipy import integrate
from dptb.data import AtomicData, AtomicDataDict

log = logging.getLogger(__name__)

try:
    import tbplas as tb
except ImportError:
    log.error('TBPLaS is not installed. Thus the ifermiaip is not available, Please install it first.')

class TBPLaS(object):
    def __init__ (
            self, 
            model: torch.nn.Module,
            results_path: Optional[str]=None,
            use_gui=False,
            overlap=False,
            device: Union[str, torch.device]=torch.device('cpu')
            ):
        
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model = model
        self.model.eval()
        self.use_gui = use_gui
        self.results_path = results_path
        self.overlap = overlap

    def get_cell(self, data: Union[AtomicData, ase.Atoms, str], AtomicData_options: dict={}):

        # get the AtomicData structure and the ase structure
        if isinstance(data, str):
            structase = read(data)
            data = AtomicData.from_ase(structase, **AtomicData_options)
        elif isinstance(data, ase.Atoms):
            structase = data
            data = AtomicData.from_ase(structase, **AtomicData_options)
        elif isinstance(data, AtomicData):
            structase = data.to_ase()
            data = data

        data = AtomicData.to_AtomicDataDict(data.to(self.device))
        data = self.model.idp(data)

        if self.overlap == True:
            assert data.get(AtomicDataDict.EDGE_OVERLAP_KEY) is not None

        # get the HR
        data = self.model(data)

        cell = data[AtomicDataDict.CELL_KEY]
        tbplus_cell = tb.PrimitiveCell(lat_vec=cell, unit=tb.ANG)

        orbs = {}
        norbs = {}
        # get_orbs
        for atomtype, orb_dict in self.model.idp.basis.items():
            split = orbs.setdefault(atomtype, [])  # split get the address of orbs's value [] of the key atomtype.
            norbs.setdefault(atomtype, 0)
            for o in orb_dict:
                if "s" in o:
                    split += [o]
                    # print(orbs[atomtype])
                    norbs[atomtype] += 1
                elif "p" in o:
                    split += [o+x for x in ["y", "z", "x"]]
                    norbs[atomtype] += 3
                elif "d" in o:
                    split += [o+x for x in ["xy ", "yz", "z2", "xz", "x2-y2"]]
                    norbs[atomtype] += 5
                else:
                    log.error("The appeared orbital is not permited in current implementation.")
                    raise RuntimeError




class _TBPLaS(object):
    def __init__(self, apiHrk, run_opt, jdata):
        self.apiH = apiHrk

        if isinstance(run_opt['structure'],str):
            self.structase = read(run_opt['structure'])
        elif isinstance(run_opt['structure'],ase.Atoms):
            self.structase = run_opt['structure']
        else:
            raise ValueError('structure must be ase.Atoms or str')
        
        self.results_path = run_opt.get('results_path')
        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)
        self.apiH.update_struct(self.structase)
        self.model_config = self.apiH.apihost.model_config
        self.jdata = jdata
 
    def write(self):
        # 1. lattice vector
        # 2. coordinates

        unit = self.apiH.unit
        if unit == "Hartree":
            factor = 13.605662285137 * 2
        elif unit == "eV":
            factor = 1.0
        elif unit == "Ry":
            factor = 13.605662285137

        lat = self.structase.cell
        tbplus_cell = tb.PrimitiveCell(lat_vec=lat, unit=tb.ANG)
        
        if os.path.exists(os.path.join(self.results_path, "HR.pth")):
            f = torch.load(os.path.join(self.results_path, "HR.pth"))
            self.all_bonds, self.hamil_blocks = f["bonds"], f["HR"]
        else:
            self.all_bonds, self.hamil_blocks, self.overlap_blocks = self.apiH.get_HR()
            assert self.overlap_blocks is None
            torch.save({"bonds":self.all_bonds, "HR":self.hamil_blocks}, os.path.join(self.results_path, "HR.pth"))

        proj_atom_anglr_m = self.apiH.structure.proj_atom_anglr_m
        orbs = {}
        norbs = {}
        # get_orbs
        for atomtype in proj_atom_anglr_m:
            split = orbs.setdefault(atomtype, [])  # split get the address of orbs's value [] of the key atomtype.
            norbs.setdefault(atomtype, 0)
            for o in proj_atom_anglr_m[atomtype]:
                if "s" in o:
                    split += [o]
                    # print(orbs[atomtype])
                    norbs[atomtype] += 1
                elif "p" in o:
                    split += [o+x for x in ["y", "z", "x"]]
                    norbs[atomtype] += 3
                elif "d" in o:
                    split += [o+x for x in ["xy ", "yz", "z2", "xz", "x2-y2"]]
                    norbs[atomtype] += 5
                else:
                    log.error("The appeared orbital is not permited in current implementation.")
                    raise RuntimeError
        # orbs for example.  {'C':[s, y, z, x, "xy ", "yz", "z2", "xz", "x2-y2"]}
        # norbs for example. {'C': 9} for use all the  s, p  and d orbitals. if only s, p {'C': 4}.

        orbsidict = {}
        # onsite part
        orbcount = 0
        elecount = 0
        R_bonds = self.all_bonds[:,4:7].abs().sum(dim=-1)
        for ix, bond in enumerate(self.all_bonds):
            itype, i, j = int(bond[0]), int(bond[1]), int(bond[3])
            # accum_norbs.append(norbs[label])
            if i == j and R_bonds[ix] < 1e-14:
                label = self.apiH.structure.proj_atom_symbols[i] # label is the atom type.
                onsite_blocks = self.hamil_blocks[ix] * factor - self.jdata.get("e_fermi", 0) # this is only for 
                assert atomic_num_dict_r[itype] == label
                norb = sum([anglrMId[''.join(re.findall(r'[A-Za-z]',ii))]*2+1 for ii in proj_atom_anglr_m[label]])
                assert norb == onsite_blocks.shape[0]
                elecount += self.jdata["nele"][label]
                for io in range(onsite_blocks.shape[0]):
                    orbsidict[str(i)+"-"+orbs[label][io]] = orbcount  # e.g.: [1-s,1-py ...]
                    orbcount += 1
                    
                    tbplus_cell.add_orbital(self.structase[i].scaled_position, 
                                            energy=onsite_blocks[io,io].item(), label=orbs[label][io])
        # accum_norbs = np.cumsum(accum_norbs)
        # off-diagonal part
        
        for ix, bond in enumerate(self.all_bonds):
            i, j, Rx, Ry, Rz = int(bond[1]), int(bond[3]), int(bond[4]), int(bond[5]), int(bond[6])
            block = self.hamil_blocks[ix] * factor
            ilabel = self.apiH.structure.proj_atom_symbols[i]
            jlabel = self.apiH.structure.proj_atom_symbols[j]
            nx, ny = len(orbs[ilabel]), len(orbs[jlabel])

            for xo in range(nx):
                for yo in range(ny):
                    energy = block[xo,yo].item()
                    idx = orbsidict[str(i)+"-"+orbs[ilabel][xo]]
                    idy = orbsidict[str(j)+"-"+orbs[jlabel][yo]]
                    if abs(energy) > 1e-7 and not \
                    ((R_bonds[ix] < 1e-14) & (idx==idy)):
                        # tbplus_cell.add_hopping(rn=[Rx, Ry, Rz], 
                        #                         orb_i=orbsidict[str(i)+"-"+orbs[ilabel][xo]],
                        #                         orb_j=orbsidict[str(j)+"-"+orbs[jlabel][yo]],
                        #                         energy=energy
                        #                         )
                        tbplus_cell._hopping_dict.add_hopping(rn=(Rx, Ry, Rz), orb_i=idx, orb_j=idy, energy=energy)
        
        if self.jdata["cal_fermi"]:
            
            nele = self.jdata["nele"]

            super_cell = tb.SuperCell(tbplus_cell, dim=self.jdata["supercell"], pbc=self.jdata["pbc"])
            sample = tb.Sample(super_cell)
            sample.rescale_ham()

            config = tb.Config()
            config.generic['nr_random_samples'] = self.jdata["nsample"]
            config.generic['nr_time_steps'] = self.jdata["ntimes"]

            solver = tb.Solver(sample, config)
            analyzer = tb.Analyzer(sample, config)

            corr_dos = solver.calc_corr_dos()
            energies, dos = analyzer.calc_dos(corr_dos)

            dos = dos / integrate.trapezoid(dos, energies)
            dos = dos * orbcount * self.jdata.get("spin_deg", 2)

            h = energies[1] - energies[0] # uniform grid
            area = (np.array(dos[:-1]) + np.array(dos[1:])) * h *  0.5
            area = np.cumsum(area)
            area = area[area<=elecount]
            e1, e2 = energies[len(area)], energies[len(area)+1]
            d1, d2 = dos[len(area)], dos[len(area)+1]
            e_fermi = e1 + (e2-e1) / (d2-d1) * (elecount-d1)
        # compute fermi-level
        else:
            e_fermi = self.jdata.get("e_fermi", 0)
        
        return tbplus_cell, e_fermi