# -*- coding: utf-8 -*-
"""
Utility functions for computing cosine angles between edge vectors and lattice vectors.
This module provides a streamlined function for neural network training workflows.
"""

import numpy as np
import torch
from pymatgen.core import Structure
from typing import Dict, Tuple


# This function remains unchanged as it's a general utility.
def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Computes cosine similarity between two sets of vectors."""
    cos_sim = torch.sum(v1 * v2, dim=-1) / (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1))
    return torch.clamp(cos_sim, -1.0, 1.0)


# This function remains unchanged.
def angle_from_array(a: np.ndarray, b: np.ndarray, lattice_matrix: np.ndarray) -> float:
    """Calculates the angle between two fractional vectors in Cartesian space."""
    a_cart = np.dot(a, lattice_matrix)
    b_cart = np.dot(b, lattice_matrix)
    cos_angle = np.dot(a_cart, b_cart) / (np.linalg.norm(a_cart) * np.linalg.norm(b_cart))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


# This function remains unchanged.
def correct_coord_sys(a: np.ndarray, b: np.ndarray, c: np.ndarray, lattice_matrix: np.ndarray) -> bool:
    """Checks if three fractional vectors form a right-handed system."""
    a_cart = np.dot(a, lattice_matrix)
    b_cart = np.dot(b, lattice_matrix)
    c_cart = np.dot(c, lattice_matrix)
    return np.dot(np.cross(a_cart, b_cart), c_cart) >= 0


# This function remains unchanged.
def same_line(a: np.ndarray, b: np.ndarray) -> bool:
    """Checks if two vectors are collinear."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return True
    cosine_val = np.dot(a, b) / (norm_a * norm_b)
    return abs(abs(cosine_val) - 1.0) < 1e-5


# This function remains unchanged.
def same_plane(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Checks if three vectors are coplanar."""
    return abs(np.dot(np.cross(a, b), c)) < 1e-5


# This function remains unchanged.
def find_local_lattice_vectors(structure: Structure) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Determines a set of three basis vectors for a local coordinate system.
    These vectors are chosen from the periodic images of an atom and are
    guaranteed to be non-collinear, non-coplanar, and form a right-handed
    system with acute angles between them.
    """
    lat_matrix = structure.lattice.matrix
    r_cut = np.max(structure.lattice.abc) + 1e-2

    neighbors = structure.get_neighbors(structure[0], r=r_cut, include_index=True)
    neighbors = sorted(neighbors, key=lambda n: n.nn_distance)

    self_images = [n.image for n in neighbors if n.index == 0 and not np.all(np.array(n.image) == 0)]

    if not self_images:
        raise ValueError("Could not find periodic images of the first atom.")

    images = np.array(self_images)

    # Find three non-coplanar vectors
    lat1 = images[0]
    lat2, lat3 = None, None

    start_idx = 1
    for i in range(start_idx, len(images)):
        if not same_line(lat1, images[i]):
            lat2 = images[i]
            start_idx = i + 1
            break
    if lat2 is None:
        raise ValueError("Could not find a second non-collinear lattice vector.")

    for i in range(start_idx, len(images)):
        if not same_plane(lat1, lat2, images[i]):
            lat3 = images[i]
            break
    if lat3 is None:
        raise ValueError("Could not find a third non-coplanar lattice vector.")

    # Ensure acute angles and a right-handed coordinate system
    if angle_from_array(lat1, lat2, lat_matrix) > 90.0:
        lat2 = -lat2
    if angle_from_array(lat1, lat3, lat_matrix) > 90.0:
        lat3 = -lat3
    if not correct_coord_sys(lat1, lat2, lat3, lat_matrix):
        lat1, lat2, lat3 = -lat1, -lat2, -lat3

    return lat1, lat2, lat3


# This function remains unchanged.
def _to_cpu_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Helper function to move tensor to CPU and convert to numpy."""
    return tensor.cpu().numpy()

# --------------------------------------------------------------
# Core function for a single structure
# --------------------------------------------------------------
def _compute_struct_cosines(
    struct_cell_np: np.ndarray,
    struct_pos_np: np.ndarray,
    struct_edge_vectors: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cosine similarities between edge vectors and the three local lattice
    vectors for a single structure.

    Args:
        struct_cell_np (np.ndarray): Lattice matrix, shape (3,3).
        struct_pos_np (np.ndarray): Cartesian coordinates of atoms, shape (N,3).
        struct_edge_vectors (torch.Tensor): Edge displacement vectors, shape (M,3).
        device (torch.device): Target device for the outputs.
        dtype (torch.dtype): Target dtype for the outputs.

    Returns:
        Tuple[torch.Tensor, torch.Tensor] or (None, None):
            - Cosine similarities, shape (M,3). Each edge vs. 3 lattice vectors.
            - Expanded lattice vectors, shape (M,3,3), repeated per edge.

        Returns (None, None) if there are no edges or atoms.
    """
    # Skip if this structure has no atoms or edges
    if struct_edge_vectors is None or struct_edge_vectors.shape[0] == 0:
        return None, None
    if struct_pos_np is None or struct_pos_np.shape[0] == 0:
        return None, None

    # Reconstruct a pymatgen Structure to determine local lattice vectors
    structure = Structure(
        lattice=struct_cell_np,
        species=["H"] * struct_pos_np.shape[0],  # Dummy species, not used
        coords=struct_pos_np,
        coords_are_cartesian=True,
    )

    # Get three local lattice directions (fractional coords)
    lat1, lat2, lat3 = find_local_lattice_vectors(structure)

    # Convert them to Cartesian coordinates
    v1 = structure.lattice.get_cartesian_coords(lat1)
    v2 = structure.lattice.get_cartesian_coords(lat2)
    v3 = structure.lattice.get_cartesian_coords(lat3)
    local_lattice_vectors_cart = np.array([v1, v2, v3])  # (3,3)

    # Convert lattice vectors to torch tensor on correct device/dtype
    lattice_vectors_tensor = torch.tensor(
        local_lattice_vectors_cart, dtype=dtype, device=device
    )

    # Expand tensors for batched cosine similarity
    edge_vectors_expanded = struct_edge_vectors.unsqueeze(1)  # (M,1,3)
    lattice_vectors_expanded = lattice_vectors_tensor.unsqueeze(0).expand(
        struct_edge_vectors.shape[0], -1, -1
    )  # (M,3,3)

    # Compute cosine similarity: each edge vs. each lattice vector
    edge_lattice_cosines = cosine_similarity(
        lattice_vectors_expanded, edge_vectors_expanded
    )

    return edge_lattice_cosines, lattice_vectors_expanded


# --------------------------------------------------------------
# Batch-aware function
# --------------------------------------------------------------
def compute_cos_angle(data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cosine similarities between edge vectors and local lattice vectors
    for a batch of structures.

    Args:
        data (Dict): Batched graph data, must contain:
            - 'pos': Atomic positions, shape (total_atoms,3).
            - 'cell': Lattice matrices, shape (batch,3,3) or (3,3) if batch=1.
            - 'edge_vectors': Edge displacement vectors, shape (total_edges,3).
            - 'batch': Atom-to-structure mapping, shape (total_atoms,).
            - 'edge_index': Graph connectivity, shape (2,total_edges).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Cosine similarities, shape (total_edges,3).
            - Expanded lattice vectors, shape (total_edges,3,3).
    """
    device = data['edge_vectors'].device
    dtype = data['edge_vectors'].dtype

    # Normalize cell shape: ensure (batch,3,3)
    cells = _to_cpu_numpy(data['cell'])
    if cells.ndim == 2:  # single structure case (3,3)
        if cells.shape == (3, 3):
            cells = cells[np.newaxis, ...]
        else:
            raise ValueError(f"Unexpected cell shape: {cells.shape}")

    # Extract atom positions and batch mapping
    positions = _to_cpu_numpy(data['pos'])             # (total_atoms,3)
    batch_atom_indices = _to_cpu_numpy(data['batch']).ravel()  # (total_atoms,)

    # Edge-related tensors stay on original device
    edge_vectors = data['edge_vectors']                # (total_edges,3)
    edge_index = data['edge_index']
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.from_numpy(edge_index).long().to(data['batch'].device)

    # Map each edge to the structure of its source atom
    edge_batch_indices = data['batch'][edge_index[0]]  # (total_edges,)

    all_cosines = []
    all_lattice_vectors = []

    num_structures = cells.shape[0]
    for i in range(num_structures):
        # Select atoms for this structure
        pos_mask = (batch_atom_indices == i)
        struct_pos = positions[pos_mask]  # numpy (n_i,3)

        # Select edges for this structure
        edge_mask = (edge_batch_indices == i)
        struct_edge_vectors = edge_vectors[edge_mask]  # torch (m_i,3)

        # Compute cosines for this structure
        cos_i, latt_i = _compute_struct_cosines(
            struct_cell_np=cells[i],
            struct_pos_np=struct_pos,
            struct_edge_vectors=struct_edge_vectors,
            device=device,
            dtype=dtype,
        )
        if cos_i is not None:
            all_cosines.append(cos_i)
            all_lattice_vectors.append(latt_i)

    # Concatenate results from all structures
    if not all_cosines:
        final_cosines = torch.empty(0, 3, device=device, dtype=dtype)
        final_lattice_vectors = torch.empty(0, 3, 3, device=device, dtype=dtype)
    else:
        final_cosines = torch.cat(all_cosines, dim=0)
        final_lattice_vectors = torch.cat(all_lattice_vectors, dim=0)

    return final_cosines, final_lattice_vectors


# UPDATED TEST CASE FOR BATCHING
def test_compute_cos_angle():
    """
    Test case for compute_cos_angle with a batch of structures to ensure it handles
    multiple structures correctly, using `edge_index` and `batch` to partition data.
    """
    # --- Structure 1: Si2 diamond structure ---
    lat1 = [[3.84, 0.0, 0.0], [0.0, 3.84, 0.0], [0.0, 0.0, 3.84]]
    spec1 = ["Si", "Si"]
    coords1 = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
    struct1 = Structure(lat1, spec1, coords1)
    # Define 2 atoms, 2 edges for structure 1
    edge_vectors1 = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], dtype=torch.float32)
    edge_index1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # Edges between atoms 0 and 1
    num_atoms1 = len(struct1)
    num_edges1 = edge_vectors1.shape[0]

    # --- Structure 2: Simple cubic Sc ---
    lat2 = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    spec2 = ["Sc"]
    coords2 = [[0.0, 0.0, 0.0]]
    struct2 = Structure(lat2, spec2, coords2)
    # Define 1 atom, 1 edge (e.g., to a periodic image) for structure 2
    edge_vectors2 = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
    edge_index2 = torch.tensor([[0], [0]], dtype=torch.long)  # Edge from atom 0 to itself
    num_atoms2 = len(struct2)
    num_edges2 = edge_vectors2.shape[0]

    # --- Batch the data together (simulating a dataloader) ---
    batched_pos = torch.cat([torch.tensor(struct1.cart_coords), torch.tensor(struct2.cart_coords)], dim=0)
    batched_cell = torch.stack([torch.tensor(struct1.lattice.matrix), torch.tensor(struct2.lattice.matrix)], dim=0)

    # Batch indices for atoms: [0, 0, 1] (2 atoms for struct 0, 1 for struct 1)
    batched_atom_indices = torch.tensor([0] * num_atoms1 + [1] * num_atoms2, dtype=torch.long)

    # Concatenate edge vectors
    batched_edge_vectors = torch.cat([edge_vectors1, edge_vectors2], dim=0)

    # Concatenate edge indices, shifting indices for the second structure
    edge_index2_shifted = edge_index2 + num_atoms1  # Shift atom indices of struct 2
    batched_edge_index = torch.cat([edge_index1, edge_index2_shifted], dim=1)

    # Test on CPU and GPU (if available)
    devices_to_test = ['cpu']
    if torch.cuda.is_available():
        devices_to_test.append('cuda')

    for device in devices_to_test:
        print(f"\n--- Testing on device: {device} ---")

        # Create the batched data dictionary on the specified device
        data = {
            'pos': batched_pos.to(device=device, dtype=torch.float32),
            'cell': batched_cell.to(device=device, dtype=torch.float32),
            'edge_vectors': batched_edge_vectors.to(device=device, dtype=torch.float32),
            'batch': batched_atom_indices.to(device=device),
            'edge_index': batched_edge_index.to(device=device),
        }

        # Run the function to be tested
        cos_angles, lattice_vectors = compute_cos_angle(data)

        # --- Verification ---
        total_edges = num_edges1 + num_edges2

        print("Test run successful!")
        assert cos_angles.device.type == device
        assert lattice_vectors.device.type == device
        print(f"Input device: {data['edge_vectors'].device}, Output device: {cos_angles.device}")

        # 1. Check output shapes
        assert cos_angles.shape == (total_edges, 3)
        assert lattice_vectors.shape == (total_edges, 3, 3)
        print(f"Output cos_angles shape: {cos_angles.shape} (Expected: ({total_edges}, 3))")
        print(f"Output lattice_vectors shape: {lattice_vectors.shape} (Expected: ({total_edges}, 3, 3))")

        # 2. Check content logic: all edges from the same structure should have the same lattice vectors
        # The first `num_edges1` rows of lattice_vectors should be identical.
        assert torch.all(lattice_vectors[0:num_edges1] == lattice_vectors[0])
        # The last `num_edges2` rows should be identical.
        if num_edges2 > 0:
            assert torch.all(lattice_vectors[num_edges1:] == lattice_vectors[num_edges1])
        # The lattice vectors for struct1 should be different from struct2.
        assert not torch.allclose(lattice_vectors[0], lattice_vectors[-1]), \
            "Lattice vectors for different structures in the batch should be different."
        print("Lattice vector batching logic is correct.")

        print(
            f"Cosine angle values (first {num_edges1} are for struct 1, next {num_edges2} for struct 2):\n{cos_angles}")

    return True


if __name__ == "__main__":
    test_compute_cos_angle()