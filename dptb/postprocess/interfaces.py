import numpy as np
import torch
import logging
import os
from typing import Optional, Union, Tuple
from ase.io import read
import ase
from dptb.data import AtomicData, AtomicDataDict
from dptb.data.interfaces.ham_to_feature import feature_to_block
from dptb.utils.constants import atomic_num_dict_r, anglrMId
from dptb.postprocess.common import load_data_for_model, get_orbitals_for_type

log = logging.getLogger(__name__)

class ToWannier90(object):
    """
    Export DeePTB model to Wannier90 format files (_hr.dat, .win, _centres.xyz)
    Compatible with TB2J, WannierBerri, etc.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            device: Union[str, torch.device] = torch.device('cpu')
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model = model
        self.model.eval()

    def _get_data_and_blocks(self, data: Union[AtomicData, ase.Atoms, str], AtomicData_options: dict = {}, e_fermi: float = 0.0):
        # Check for overlap
        if getattr(self.model, "overlap", False):
            raise ValueError("Export to Wannier90 format does not support models with non-orthogonal bases (overlap). Please use an orthogonal model.")

        # Use centralized data loading
        data = load_data_for_model(
            data=data,
            model=self.model,
            device=self.device,
            AtomicData_options=AtomicData_options
        )
        self.positions = data[AtomicDataDict.POSITIONS_KEY].numpy()
        self.scaled_pos = data[AtomicDataDict.POSITIONS_KEY] @  data[AtomicDataDict.CELL_KEY].inverse()
        self.scaled_pos = self.scaled_pos.numpy()
        self.atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().cpu().numpy()
        self.symbols = [self.model.idp.type_names[t] for t in self.atom_types]
        self.cell = data[AtomicDataDict.CELL_KEY].numpy()
        data = self.model(data)
        
        blocks = feature_to_block(data, idp=self.model.idp)
        return data, blocks

    def write_hr(self, data: Union[AtomicData, ase.Atoms, str], filename: str = "wannier90_hr.dat", AtomicData_options: dict = {}, e_fermi: float = 0.0):
        """
        Write the Hamiltonian to Wannier90 _hr.dat format.
        """
        data_dict, blocks = self._get_data_and_blocks(data, AtomicData_options, e_fermi)
        
        # 1. Count orbitals and build map
        atom_types = data_dict[AtomicDataDict.ATOM_TYPE_KEY].flatten().cpu().numpy()
        num_atoms = len(atom_types)
        
        # Build orbital map: (atom_idx, orb_name) -> global_idx (0-based)
        # Assuming order is by atom index, then by orbital definition in model.idp
        # Reconstruct orbital list per atom to ensure consistent ordering
        # Using similar logic to totbplas.py to determine orbital count/order
        orbs_per_type = {}
        for atomtype, orb_dict in self.model.idp.basis.items():
            orb_list = get_orbitals_for_type(orb_dict)
            #orb_list = []
            #for o in orb_dict:
            #    if "s" in o: orb_list.append(o)
            #    elif "p" in o: orb_list.extend([o+"_y", o+"_z", o+"_x"]) # Standard Wannier90 p-order usually z,x,y or similar? keeping dptb order
            #    elif "d" in o: orb_list.extend([o+"_xy", o+"_yz", o+"_z2", o+"_xz", o+"_x2-y2"])
            orbs_per_type[atomtype] = orb_list

        global_idx = 0
        atom_orb_start = [] # Starting global index for each atom
        for i in range(num_atoms):
            itype = atom_types[i]
            isymbol = self.model.idp.type_names[itype]
            atom_orb_start.append(global_idx)
            global_idx += len(orbs_per_type[isymbol])
        num_wann = global_idx
        
        # 2. Collect hoppings
        # blocks keys are "i_j_Rx_Ry_Rz"
        # We need to restructure this into (Rx, Ry, Rz) -> Matrix[num_wann, num_wann]
        # To handle arbitrary R vectors efficiently, we can use a dictionary
        # R_dict[(rx, ry, rz)] = np.zeros((num_wann, num_wann), dtype=complex)
        from collections import defaultdict
        R_dict = defaultdict(lambda: np.zeros((num_wann, num_wann), dtype=complex))
        
        # Process blocks
        for bond_key, block_tensor in blocks.items():
            # bond_key format: "i_j_Rx_Ry_Rz"
            parts = bond_key.split('_')
            i_atom = int(parts[0])
            j_atom = int(parts[1])
            R = tuple(map(int, parts[2:])) # (Rx, Ry, Rz)
            block_np = block_tensor.detach().cpu().numpy()
            # Subtract Fermi energy from onsite terms (R=0, i=j)
            if R == (0,0,0) and i_atom == j_atom:
                block_np = block_np - e_fermi * np.eye(block_np.shape[0])        
            start_i = atom_orb_start[i_atom]
            start_j = atom_orb_start[j_atom]
            end_i = start_i + block_np.shape[0]
            end_j = start_j + block_np.shape[1]
            
            # 3. Add H(-R) = H(R)^dag
            # Check if we are handling onsite or hopping
            # If onsite (R=0, i=j), H(0) must be Hermitian. H(0)^dag = H(0).
            # But feature_to_block for onsite might not double count? 
            # Actually for onsite, i=j, R=0. feature_to_block just accumulates.
            # If we divided by 2, we need to add transpose?
            # Wait, onsite terms (R=0, i=j) are self-conjugate.
            # Let's handle R=0 i=j separately to be safe from double counting if logic differs.
            # 1. onsite block
            if R == (0,0,0) and i_atom == j_atom:                
                R_dict[R][start_i:end_i, start_j:end_j] += block_np
                # No -R to add, R=-R=0
            # Hopping blocks (including 0,0,0, i \neq j)
            else:
                # i,j,+R
                R_dict[R][start_i:end_i, start_j:end_j] += block_np
                # j,i,-R
                # import!!! tuple(-x for x in [0,0,0]) = (0,0,0) ! no -0 This is important!! 
                neg_R = tuple(-x for x in R)
                # Swap indices for conjugate block: H(-R) maps j -> i
                # H(-R) shape should be (n_orb_j, n_orb_i)
                # block_np^dag shape is (cols, rows) -> (n_orb_j, n_orb_i). Correct.
                
                start_i_rev = atom_orb_start[j_atom]
                start_j_rev = atom_orb_start[i_atom]
                end_i_rev = start_i_rev + block_np.shape[1]
                end_j_rev = start_j_rev + block_np.shape[0]
                R_dict[neg_R][start_i_rev:end_i_rev, start_j_rev:end_j_rev] += block_np.conj().T
            
        # 3. Write file
        # Sort R vectors to ensure deterministic output (and usually 0 0 0 first)
        sorted_keys = sorted(R_dict.keys(), key=lambda x: (x[0]**2+x[1]**2+x[2]**2, x[2], x[1], x[0]))
        nrpts = len(sorted_keys)
        
        with open(filename, 'w') as f:
            f.write(f" written by DeePTB export\n")
            f.write(f"{num_wann}\n")
            f.write(f"{nrpts}\n")
            
            # Wigner-Seitz degeneracies - assume 1 for now
            # Format: 15 integers per line
            degen = [1] * nrpts
            for i in range(0, nrpts, 15):
                line = "    ".join(map(str, degen[i:min(i+15, nrpts)]))
                f.write(f"    {line}\n")
            
            # Write Hamiltonian elements
            # Format: Rx Ry Rz m n Re[H] Im[H]
            # m, n are 1-based indices of Wannier functions
            # Loops: R, n (col), m (row)  <-- check specific convention, usually standard Wannier90 is:
            # for iR in range(nrpts):
            #   for n in range(num_wann):
            #     for m in range(num_wann):
            #       ...
            for R in sorted_keys:
                H_R = R_dict[R]
                rx, ry, rz = R
                for n in range(num_wann): # 0-based
                    for m in range(num_wann): # 0-based
                        val = H_R[m, n] # row m, col n
                        f.write(f"    {rx:5d}    {ry:5d}    {rz:5d}    {m+1:5d}    {n+1:5d}    {val.real:20.12f}    {val.imag:20.12f}\n")
                        
        log.info(f"Wrote Wannier90 Hamiltonian to {filename}")

    def write_win(self, data: Union[AtomicData, ase.Atoms, str], filename: str = "wannier90.win", e_fermi: float = 0.0):
        """
        Write a minimal .win file for TB2J/WannierBerri compatibility.
        """        
        # Calculate num_wann
        # Reusing logic from write_hr, maybe refactor if strict modularity needed
        orbs_per_type = {}
        for atomtype, orb_dict in self.model.idp.basis.items():
            count = 0
            for o in orb_dict:
                if "s" in o: count += 1
                elif "p" in o: count += 3
                elif "d" in o: count += 5
            orbs_per_type[atomtype] = count
            
        num_wann = 0
        for itype in self.atom_types:
            isymbol = self.model.idp.type_names[itype]
            num_wann += orbs_per_type[isymbol]
    
        
        with open(filename, 'w') as f:
            f.write("! Win file generated by DeePTB\n\n")
            f.write(f"num_bands = {num_wann}\n")
            f.write(f"num_wann = {num_wann}\n\n")
            
            f.write("begin unit_cell_cart\n")
            # f.write("Ang\n") # PythTB might NOT support unit specifier line, usually defaults to Angstrom or reads raw numbers.
            # Removing unit line to test compatibility.
            for vec in self.cell:
                f.write(f" {vec[0]:12.8f} {vec[1]:12.8f} {vec[2]:12.8f}\n")
            f.write("end unit_cell_cart\n\n")
            
            f.write("begin atoms_frac\n")
            for s, p in zip(self.symbols, self.scaled_pos):
                f.write(f"{s}  {p[0]:12.8f} {p[1]:12.8f} {p[2]:12.8f}\n")
            f.write("end atoms_frac\n")
            
        log.info(f"Wrote minimal Wannier90 win file to {filename}")

    def write_centres(self, data: Union[AtomicData, ase.Atoms, str], filename: str = "wannier90_centres.xyz"):
        """
        Write centres file (often approximated as atom positions for atomic orbitals).
        Format: atom_symbol X Y Z (Cartesian Angstrom)
        TB2J often expects this for reading positions.
        """        
        # In LCAO/TB, "Wannier centers" are usually just the atomic positions + orbital offsets
        # For simplicity, we write atomic positions repeated for each orbital, or just atomic positions?
        # Wannier90 *_centres.xyz usually lists ALL Wannier centers (num_wann lines).
        # Need to iterate atoms and their orbitals
        centres = []
        orbs_per_type_list = {}
        for atomtype, orb_dict in self.model.idp.basis.items():
            orb_list = []
            for o in orb_dict:
                if "s" in o: orb_list.append(o)
                elif "p" in o: orb_list.extend([o+"_y", o+"_z", o+"_x"]) 
                elif "d" in o: orb_list.extend([o+"_xy", o+"_yz", o+"_z2", o+"_xz", o+"_x2-y2"])
            orbs_per_type_list[atomtype] = orb_list
            
        for i in range(len(self.atom_types)):
            itype = self.atom_types[i]
            isymbol = self.model.idp.type_names[itype]
            pos = self.positions[i]
            
            # For each orbital on this atom, add a centre
            # Ideally accurate centers (e.g. p-orbitals are centered on atom), so this is fine.
            num_orbs = len(orbs_per_type_list[isymbol])
            for _ in range(num_orbs):
            # Using 'X' as generic element for center, or the atom symbol?
                # Wannier90 output usually has "X" or element symbol.
                # WARNING: PythTB's w90 reader STRICTLY expects the symbol to be "X".
                # If it finds "Si" or anything else, it raises "Inconsistency in the centres file."
                # To maintain compatibility with PythTB, we force "X".
                centres.append(("X", pos))
                
        with open(filename, 'w') as f:
            f.write(f"{len(centres)}\n")
            f.write("Wannier centres generated by DeePTB\n")
            for sym, pos in centres:
                f.write(f"{sym} {pos[0]:12.8f} {pos[1]:12.8f} {pos[2]:12.8f}\n")
                
        log.info(f"Wrote Wannier centres to {filename}")


class ToPythTB(object):
    """
    Convert DeePTB model to PythTB model object.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            device: Union[str, torch.device] = torch.device('cpu')
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model = model
        self.model.eval()
        
        try:
            from pythtb import tb_model
        except ImportError:
            log.exception("PythTB not installed. Run `pip install pythtb`")
            raise

    def get_model(self, data: Union[AtomicData, ase.Atoms, str], AtomicData_options: dict = {}, e_fermi: float = 0.0):
        from pythtb import tb_model
        
        # Check for overlap
        if getattr(self.model, "overlap", False):
            raise ValueError("Export to PythTB does not support models with non-orthogonal bases (overlap). Please use an orthogonal model.")

        # Use centralized data loading
        data = load_data_for_model(
            data=data,
            model=self.model,
            device=self.device,
            AtomicData_options=AtomicData_options
        )
        
        # Run model forward
        data = self.model(data)
        blocks = feature_to_block(data, idp=self.model.idp)
        
        # 1. Setup PythTB model
        lat_vecs = data['cell'].cpu().numpy()
        positions = data['pos'].cpu().numpy()
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().cpu().numpy()
        
        # Prepare orbitals
        orbs_per_type_list = {}
        for atomtype, orb_dict in self.model.idp.basis.items():
            orb_list = get_orbitals_for_type(orb_dict)
            #orb_list = []
            #for o in orb_dict:
            #    if "s" in o: orb_list.append(o)
            #    elif "p" in o: orb_list.extend([o+"_y", o+"_z", o+"_x"]) 
            #    elif "d" in o: orb_list.extend([o+"_xy", o+"_yz", o+"_z2", o+"_xz", o+"_x2-y2"])
            orbs_per_type_list[atomtype] = orb_list
            
        orb_coords = []
        # PythTB needs orbital coordinates. 
        # If lat_vecs is provided, PythTB expects Fractional coordinates.
        # Reference: PythTB documentation and totbplas.py implementation.
        
        # Convert positions to fractional
        try:
            inv_lat = np.linalg.inv(lat_vecs)
            # positions shape (N, 3). lat_vecs shape (3, 3) (rows are vectors)
            # pos = frac @ lat_vecs  =>  frac = pos @ inv_lat
            frac_positions = positions @ inv_lat
        except np.linalg.LinAlgError:
            log.warning("Lattice vectors singular, using Cartesian for PythTB (may be incorrect if not 3D)")
            frac_positions = positions

        for i in range(len(atom_types)):
            itype = atom_types[i]
            isymbol = self.model.idp.type_names[itype]
            # Use fractional position for orbital center
            pos = frac_positions[i]
            
            for _ in orbs_per_type_list[isymbol]:
                orb_coords.append(pos)
                
        # Create model object
        my_model = tb_model(3, 3, lat_vecs, orb_coords)
        
        # 2. Add hoppings
        # We need a mapping from (atom_idx, local_orb_idx) to global_orb_idx
        
        atom_orb_start = []
        global_idx = 0
        for i in range(len(atom_types)):
            itype = atom_types[i]
            isymbol = self.model.idp.type_names[itype]
            atom_orb_start.append(global_idx)
            global_idx += len(orbs_per_type_list[isymbol])

        for bond_key, block_tensor in blocks.items():
            parts = bond_key.split('_')
            i_atom = int(parts[0])
            j_atom = int(parts[1])
            R = tuple(map(int, parts[2:]))
            
            block_np = block_tensor.detach().cpu().numpy()
            
            if R == (0,0,0) and i_atom == j_atom:
                # Onsite
                block_np = block_np - e_fermi * np.eye(block_np.shape[0])
                # Set onsite energies for PythTB
                start_i = atom_orb_start[i_atom]
                start_j = atom_orb_start[j_atom]
                rows, cols = block_np.shape
                 
                # Iterate diagonal for onsite energies
                for orb_idx in range(rows):
                    val = block_np[orb_idx, orb_idx]
                    global_idx_onsite = start_i + orb_idx
                    my_model.set_onsite(val.real, global_idx_onsite, mode="reset")
                     
                # Off-diagonal within onsite block
                for r in range(rows):
                    for c in range(cols):
                        if r == c: continue 
                        val = block_np[r, c]
                        if abs(val) > 1e-10:
                            idx_i = start_i + r
                            idx_j = start_j + c
                            
                            # Filter conjugates: only keep canonical
                            # key = (R, i, j) vs (-R, j, i)
                            # tuple comparison: (0,0,0, idx_i, idx_j) vs (0,0,0, idx_j, idx_i)
                            # For R=0: keep if idx_i < idx_j
                            if idx_i < idx_j:
                                my_model.set_hop(val, idx_i, idx_j, list(R), mode="reset")

            else:
                # Hopping block (R!=0 or i!=j)
                # feature_to_block sums the bidirectional edges (i->j and j->i), 
                # effectively resulting in 2 * Energy for Hermitian systems.
                # Reference totbplas.py averages them: (E + E_rev)/2.
                # So we must divide by 2 here to get the correct single-direction hopping amplitude.
                block_np = block_np

                start_i = atom_orb_start[i_atom]
                start_j = atom_orb_start[j_atom]
                
                rows, cols = block_np.shape
                for r in range(rows):
                    for c in range(cols):
                        val = block_np[r, c]
                        if abs(val) > 1e-10:
                            idx_i = start_i + r
                            idx_j = start_j + c
                            
                            # feature_to_block ensures unique bond blocks (i <= j, and R/ -R combined).
                            # So we do not need to check for conjugates here,
                            # except that PythTB automatically handles the hermitian conjugate.
                            # Since feature_to_block gives us one unique block for the interaction,
                            # we simply add all non-zero elements.
                            my_model.set_hop(val, idx_i, idx_j, list(R), mode="reset",allow_conjugate_pair=True)
                        


        return my_model
