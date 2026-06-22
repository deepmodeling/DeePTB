import torch
from typing import Tuple
from dptb.utils.constants import Bohr2Ang, Harte2eV
from dptb.data import  AtomicDataDict
from dptb.data.transforms import OrbitalMapper
from dptb.nn.dftb.sk_param import SKParam
from dptb.utils.ewald_sum import build_coulomb_matrix, Geometry_

def calculate_expgamma(
    atom_us: torch.Tensor,
    edge_lengths: torch.Tensor,
    edge_atom_types: torch.Tensor,
    node_atom_types: torch.Tensor,
    minvalues: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:    
    """
    Calculate the exponential gamma function for DFTB calculations.
    
    Parameters
    ----------
    atom_us : torch.Tensor
        Tensor containing the atom-resolved Hubbard U values for each atom type.
        Shape should be [num_atom_types, 1, 1].
    edge_lengths : torch.Tensor
        Tensor containing the lengths of edges between atoms.
        Shape should be [num_edges].
    edge_atom_types : torch.Tensor
        Tensor containing the atom types for each edge.
        Shape should be [2, num_edges].
    node_atom_types : torch.Tensor
        Tensor containing the atom types for each node.
        Shape should be [num_atom_type, 1].
    minvalues : float, optional
        Minimum threshold value to prevent numerical instabilities, by default 1e-6.
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing (expGamma, expGamma_onsite), where expGamma is the
        exponential gamma for edges and expGamma_onsite is for onsite terms.
        Shape of expGamma is [num_edges] and expGamma_onsite is [num_atoms].
        
    Raises
    ------
    ValueError
        If any Hubbard U value is too small (less than minvalues).
    AssertionError
        If any edge length is negative or zero (less than or equal to minvalues).
    """
    # Convert inputs to float64 for numerical precision in SCC calculations
    atom_us_dp = atom_us.to(dtype=torch.float64, device=edge_lengths.device)
    edge_lengths_dp = edge_lengths.to(dtype=torch.float64)
    edge_atom_types = edge_atom_types.to(device=edge_lengths.device)
    node_atom_types = node_atom_types.to(device=edge_lengths.device)

    # Get Hubbard U values for each atom in the edges
    Ua = atom_us_dp[edge_atom_types[0]].reshape(-1) # shape: [num_edges]
    Ub = atom_us_dp[edge_atom_types[1]].reshape(-1) # shape: [num_edges]

    # Check for very small U values which could cause numerical issues
    err_index = torch.where(atom_us_dp.flatten() < minvalues)[0]
    if len(err_index) > 0:
        raise ValueError("Failure in short-range gamma, U too small for atom number")

    # Check for negative or zero distances
    assert (edge_lengths_dp > minvalues).all(), "Failure in short-range gamma, r_ab negative or zero"


    # Initialize expGamma for each edge
    expGamma = torch.zeros([edge_atom_types.shape[1]], dtype=torch.float64, device=edge_lengths_dp.device) # shape: [num_edges]
    
    # Identify edges where Ua ≈ Ub (same atom type or very close U values)
    equ_u_mask = torch.abs(Ua - Ub) < minvalues
    
    # The formula is expressed in Atomic Units (Energy in Hartree, Distance in Bohr)
    # Calculate expGamma for edges with equal U values
    if equ_u_mask.any():
        tauMean = 0.5 * 16./5. * (Ua[equ_u_mask] + Ub[equ_u_mask]) / Harte2eV  # to Hartree
        rab_equU =  edge_lengths_dp[equ_u_mask] / Bohr2Ang # to Bohr

        expGamma[equ_u_mask] =  torch.exp(-tauMean * rab_equU) * (
            48.0 / rab_equU +
            33.0 * tauMean +
            9.0 * rab_equU * (tauMean ** 2) +
            1.0 * (rab_equU ** 2) * (tauMean ** 3)
        ) / 48.0

    # Calculate expGamma for edges with different U values
    if (~equ_u_mask).any():
        tauA = 16./5. * Ua[~equ_u_mask] / Harte2eV  # to Hartree
        tauB = 16./5. * Ub[~equ_u_mask] / Harte2eV  # to Hartree
        rab_nequU =  edge_lengths_dp[~equ_u_mask] / Bohr2Ang # to Bohr
        
        part1 = torch.exp(-tauA * rab_nequU) * (
            0.5 * tauB ** 4 * tauA / (tauA ** 2 - tauB ** 2) ** 2 - 
            (tauB ** 6 - 3.0 * tauB ** 4 * tauA ** 2) / (rab_nequU * (tauA ** 2 - tauB ** 2) ** 3)
        )
        
        part2 = torch.exp(-tauB * rab_nequU) * (
            0.5 * tauA ** 4 * tauB / (tauA ** 2 - tauB ** 2) ** 2 -
            (tauA ** 6 - 3.0 * tauA ** 4 * tauB ** 2) / (rab_nequU * (tauB ** 2 - tauA ** 2) ** 3)
        )
        
        expGamma[~equ_u_mask] = (part1 + part2)
    
    # Convert expGamma from Hartree to eV
    expGamma = expGamma * Harte2eV
    # unit: eV
    expGamma_onsite = atom_us_dp[node_atom_types.flatten()].reshape(-1) # shape: [num_atom_types]

    return expGamma, expGamma_onsite



def get_expgamma(
                data: AtomicDataDict,
                idp: OrbitalMapper,
                skp:  SKParam) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the expGamma and expGamma_onsite tensors for a given atomic data input.
    Args:
        data (AtomicDataDict): Dictionary containing atomic data, including edge types, edge lengths, and atom types.
        idp (OrbitalMapper): Mapper object for transforming and untransforming bond and atom indices.
        skp (SKParam or SCCParams): Object containing atom-resolved Hubbard U.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - expGamma: Tensor of expGamma values for each edge.
            - expGamma_onsite: Tensor of expGamma values for onsite interactions.
    """      
    # Calculate expGamma values
    edge_index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten()
    edge_number = idp.untransform_bond(edge_index).T
    edge_atom_types = idp.transform_atom(edge_number.flatten()).reshape(2, -1)

    atom_u = skp.skdict.get("Atom_U", skp.skdict.get("Highest_Occu_U"))
    expGamma, expGamma_onsite = calculate_expgamma( atom_us=atom_u,
                                                    edge_lengths=data[AtomicDataDict.EDGE_LENGTH_KEY],
                                                    edge_atom_types=edge_atom_types,
                                                    node_atom_types=data[AtomicDataDict.ATOM_TYPE_KEY])
    return expGamma, expGamma_onsite


def get_inv_r(data: AtomicDataDict) -> torch.Tensor:

    assert 'cell' in data.keys(), "Cell information is missing in data."
    assert 'pos' in data.keys(), "Position information is missing in data."
    system = Geometry_(data)
    inv_r = build_coulomb_matrix(system) # unit: 1/Bohr
    inv_r = inv_r.to(dtype=torch.float64) * Harte2eV # unit: eV, converted to float64
    return inv_r


def get_Gamma(data: AtomicDataDict,
              expGamma: torch.Tensor,
              expGamma_onsite: torch.Tensor,
              inv_r: torch.Tensor) -> torch.Tensor:
    
    assert AtomicDataDict.EDGE_CELL_SHIFT_KEY in data, "edge_cell_shift key not found in data."
    assert AtomicDataDict.ATOM_TYPE_KEY in data, "atom_types key not found in data."

    n_atoms = data[AtomicDataDict.ATOM_TYPE_KEY].shape[0]
    edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]  # [2, n_edges]
    shifts = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY] # [n_edges, 3]


    unique_shifts, shift_ids = torch.unique(shifts, dim=0, return_inverse=True)
    n_shifts = unique_shifts.shape[0]
    expGamma_block = torch.zeros((n_shifts, n_atoms, n_atoms), dtype=expGamma.dtype, device=expGamma.device)

    # fill expGamma in the block matrix
    # [n_edges, 1] -> (shift_id, iatom, jatom)
    iatoms, jatoms = edge_index
    expGamma_block.index_put_((shift_ids, iatoms, jatoms), expGamma, accumulate=False)

    # onsite term for zero shift
    zero_shift_mask = (unique_shifts == 0).all(dim=1)
    if zero_shift_mask.any():
        zero_idx = torch.nonzero(zero_shift_mask, as_tuple=False).item()
        diag_idx = torch.arange(n_atoms, device=expGamma.device)
        expGamma_block[zero_idx, diag_idx, diag_idx] = -expGamma_onsite

    # summation for Gamma
    Gamma = inv_r - expGamma_block.sum(dim=0)
    assert torch.allclose(Gamma, Gamma.T, atol=1e-8), "Gamma matrix is not symmetric."

    return Gamma



# def get_Gamma(  data: AtomicDataDict,
#                 expGamma: torch.Tensor,
#                 expGamma_onsite: torch.Tensor,
#                 inv_r: torch.Tensor) -> torch.Tensor:
    
#     assert AtomicDataDict.EDGE_CELL_SHIFT_KEY in data.keys(), "edge_cell_shift key not found in data."
#     assert AtomicDataDict.ATOM_TYPE_KEY in data.keys(), "atom_types key not found in data."
            
#     unique_shifts = torch.unique(data[AtomicDataDict.EDGE_CELL_SHIFT_KEY], dim=0)

#     expGamma_block = torch.zeros((len(unique_shifts), 
#                                     data[AtomicDataDict.ATOM_TYPE_KEY].shape[0], 
#                                     data[AtomicDataDict.ATOM_TYPE_KEY].shape[0]))
#     for idx, shift in enumerate(unique_shifts):
#         mask = (data[AtomicDataDict.EDGE_CELL_SHIFT_KEY]==shift).all(dim=1)
#         ist = 0
#         for iatom, jatom in data[AtomicDataDict.EDGE_INDEX_KEY][:,mask].T:
#             expGamma_block[idx][iatom,jatom] = expGamma[mask][ist]
#             ist += 1
#         if torch.all(shift == 0):
#             for diag_i in range(data[AtomicDataDict.ATOM_TYPE_KEY].shape[0]):
#                 expGamma_block[idx][diag_i, diag_i] = -1 * expGamma_onsite[diag_i]

#     Gamma = inv_r - expGamma_block.sum(dim=0) # unit: eV
#     assert torch.allclose(Gamma, Gamma.T, atol=1e-8), "Gamma matrix is not symmetric."

#     return Gamma
