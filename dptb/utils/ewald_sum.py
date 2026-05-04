# -*- coding: utf-8 -*-

"""
This function is adapted from TBMaLT
GitHub: https://github.com/tbmalt/tbmalt
License: GNU Lesser General Public License v3 (LGPL-3.0-or-later)

Original copyright:
TBMaLT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

Modifications in this version:
- [Describe your changes here, e.g., adjusted arguments / optimized loop]
"""


"""Code associated with coulombic interactions.

This module calculate the Ewald summation for periodic
boundary conditions.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List, Any
import torch
import numpy as np
from scipy import special
from dptb.utils.constants import Bohr2Ang, euler
from dptb.data import AtomicDataDict

Tensor = torch.Tensor
Sliceable = Union[List[Tensor], Tuple[Tensor]]
# Todo:
#   - Currently, the `Ewald._update_latvec` and `Ewald._update_neighbour`
#     methods are very wasteful. This is because they do not take advantage of
#     any caching. They instead recompute everything from scratch each time
#     they are called. They should be rewritten to take advantage of caching.

class Geometry_(object):

    def __init__(self, data):

        assert AtomicDataDict.CELL_KEY in data.keys(), "Cell information is missing in data."
        assert AtomicDataDict.POSITIONS_KEY in data.keys(), "Position information is missing in data."

        self.lattice: Tensor = data[AtomicDataDict.CELL_KEY] / Bohr2Ang 
        # self.n_atoms: Tensor = atomic_numbers.count_nonzero(-1)
        self.positions: Tensor = data[AtomicDataDict.POSITIONS_KEY] / Bohr2Ang
        self.n_atoms : Tensor = torch.tensor(len(data[AtomicDataDict.POSITIONS_KEY]))
        self.periodicity = Periodicity_(data)
    
    @property
    def pbc(self):
        return self.periodicity.pbc

class Periodicity_(object):

    def __init__(self, data):

        assert AtomicDataDict.CELL_KEY in data.keys(), "Cell information is missing in data."

        self.lattice = data[AtomicDataDict.CELL_KEY] / Bohr2Ang
        self.reciprocal_lattice = torch.from_numpy(
            2 * np.pi * np.mat(self.lattice.cpu().numpy()).I.T
            ).to(self.lattice.dtype).to(self.lattice.device)
        self.pbc = self.lattice.ne(0).any(-1)

    @property
    def cellvol(self) -> Tensor:
        """Get unit cell volume."""
        return abs(torch.det(self.lattice))
    
    @staticmethod
    def inverse_lattice_vector(lattice_vector: Tensor) -> Tensor:
        """Calculate the inverse of lattice vectors in 3D case."""
        eye = torch.eye(
                lattice_vector.shape[-1], device=lattice_vector.device,
                dtype=lattice_vector.dtype)
        
        invlat = torch.linalg.solve(lattice_vector, eye).transpose(-1, -2)
        return invlat

    @staticmethod
    def get_cell_translation_vector_indices(
            inverse_lattice_vector: Tensor, cutoff,
            **kwargs):
        """Calculate cell translation vector indices."""

        # This is exposed as a static method for the sake of the coulomb module

        # Issues, will change with the cutoff value thus this value should be
        # static.

        dd = {'dtype': inverse_lattice_vector.dtype,
              'device': inverse_lattice_vector.device}

        # Ranges of cell translation on three dimensions
        n_images = bT(torch.ceil(cutoff * bT(torch.norm(
            inverse_lattice_vector, dim=-1))))

        ranges = torch.stack([-n_images, n_images])

        # For 1D/2D cell translation, non-periodic direction will be zero
        mask_zero = kwargs.get(
            'mask_zero', inverse_lattice_vector.eq(0).all(-1))
        ranges[torch.stack([mask_zero, mask_zero])] = 0

        # length of the first, second and third column in ranges
        leng = ranges[1, :].long() - ranges[0, :].long() + 1

        # number of cells
        n_cells = leng[..., 0] * leng[..., 1] * leng[..., 2]

        # Cell translation vectors in relative coordinates
        if inverse_lattice_vector.ndim == 2:  # -> single
            # Would be worth replacing this with `torch.cartesian_prod` at
            # some point.
            cellvec = torch.stack([
                torch.linspace(ranges[0, 0], ranges[1, 0],
                               leng[0], **dd).repeat_interleave(leng[2] * leng[1]),
                torch.linspace(ranges[0, 1], ranges[1, 1],
                               leng[1], **dd).repeat(leng[0]).repeat_interleave(leng[2]),
                torch.linspace(ranges[0, 2], ranges[1, 2],
                               leng[2], **dd).repeat(leng[0] * leng[1])])

        else:  # -> batch
            # Large values are padded at the end of short cell vectors.
            cellvec = pack([torch.stack([
                torch.linspace(iran[0, 0], iran[1, 0],
                               ile[0], **dd).repeat_interleave(ile[2] * ile[1]),
                torch.linspace(iran[0, 1], iran[1, 1],
                               ile[1], **dd).repeat(ile[0]).repeat_interleave(ile[2]),
                torch.linspace(iran[0, 2], iran[1, 2],
                               ile[2], **dd).repeat(ile[0] * ile[1])])
                for ile, iran in zip(
                    leng, ranges.transpose(-2, -3))], value=1e4)

        return cellvec.transpose(-1, -2), n_cells
    

    @staticmethod
    def get_cell_translation_vectors(
            lattice_vector: Tensor, cell_translation_vector_indices: Tensor):
        """Cell translation vectors in absolute units."""
        return (
            torch.matmul(lattice_vector.transpose(-1, -2),
                         cell_translation_vector_indices.transpose(-1, -2))
        ).transpose(-1, -2)

    @staticmethod
    def get_periodic_distances(
            cell_translation_vectors: Tensor, positions: Tensor, n_atoms=None,
            **kwargs):
        """Distances between atoms in the central and neighbouring cells."""

        # Positions of atoms in all images
        positions_expanded = (cell_translation_vectors.unsqueeze(-2) +
                     positions.unsqueeze(-3))

        # Distance matrix, large values will be padded for batch systems
        if positions.ndim == 2:  # -> single
            distance = torch.sqrt(
                ((positions_expanded.repeat(1, n_atoms, 1) - torch.repeat_interleave(
                    positions, n_atoms, 0)) ** 2).sum(-1).reshape(
                    -1, n_atoms, n_atoms))

        else:  # -> batch
            if n_atoms is None:
                raise ValueError(
                    "Number of atoms per system must be specified when working"
                    " with batches as it cannot be inferred easily.")
            distance = pack(
                [torch.sqrt(((ipos[:, :inat].repeat(1, inat, 1) -
                              icp[:inat].repeat_interleave(inat, 0)
                              ) ** 2).sum(-1)).reshape(-1, inat, inat)
                 for ipos, icp, inat in zip(
                    positions_expanded, positions, n_atoms)
                 ], value=1e3)

        return distance

    @staticmethod
    def get_neighbours(periodic_distances, cutoff) -> Tensor:
        """A mask to choose atoms of images inside the cutoff distance."""
        if len(cutoff) == 1:
            return periodic_distances.le(cutoff)
        else:
            return torch.stack([ipd.le(ico) for ipd, ico in zip(
                periodic_distances, cutoff)])




def bT(tensor: Tensor) -> Tensor:
    """Dimensionally agnostic "transpose".

    Reverses the dimensions of a tensor like so [m, n, o] -> [o, n, m]. This is
    designed to preserve the original functionality of the `torch.T` operator
    in an effort to maintain dimensional/batch agnosticism. Recent versions of
    PyTorch will only permit the transpose operator to be used on 2D matrices
    which makes dimensionally agnostic treatment of tensors difficult in some
    situations.

    Arguments:
        tensor: the tensor whose dimensions are to be flipped.

    Returns:
        flipped_tensor: the tensor with its dimensions reversed.

    """
    return tensor.permute(*torch.arange(tensor.ndim - 1, -1, -1))

def pack(tensors: Sliceable, axis: int = 0,
         value: Any = 0, size: Optional[Union[Tuple[int], torch.Size]] = None,
         return_mask: bool = False) -> Union[Tensor, Optional[Tensor]]:
    """Pad and pack a sequence of tensors together.

    Pad a list of variable length tensors with zeros, or some other value, and
    pack them into a single tensor.

    Arguments:
        tensors: List of tensors to be packed, all with identical dtypes.
        axis: Axis along which tensors should be packed; 0 for first axis -1
            for the last axis, etc. This will be a new dimension. [DEFAULT=0]
        value: The value with which the tensor is to be padded. [DEFAULT=0]
        size: Size of each dimension to which tensors should be padded. This
            defaults to the largest size encountered along each dimension.
        return_mask: If True, a mask identifying the padding values is
            returned. [DEFAULT=False]

    Returns:
        packed_tensors: Input tensors padded and packed into a single tensor.
        mask: A tensor that can mask out the padding values. A False value in
            ``mask`` indicates the corresponding entry in ``packed_tensor`` is
            a padding value.

    Notes:
        ``packed_tensors`` maintains the same order as ``tensors``. This
        is faster & more flexible than the internal pytorch pack & pad
        functions (at this particular task).

        If ``tensors`` is a `torch.tensor` it will be immedatly returned. This
        helps with batch agnostic programming.

    Examples:
        Multiple tensors can be packed into a single tensor like so:

        >>> from tbmalt.common.batch import pack
        >>> import torch
        >>> a, b, c = torch.rand(2,2), torch.rand(3,3), torch.rand(4,4)
        >>> abc_packed_a = pack([a, b, c])
        >>> print(abc_packed_a.shape)
        torch.Size([3, 4, 4])
        >>> abc_packed_b = pack([a, b, c], axis=1)
        >>> print(abc_packed_b.shape)
        torch.Size([4, 3, 4])
        >>> abc_packed_c = pack([a, b, c], axis=-1)
        >>> print(abc_packed_c.shape)
        torch.Size([4, 4, 3])

        An optional mask identifying the padding values can also be returned:

        >>> packed, mask = pack([torch.tensor([1.]),
        >>>                      torch.tensor([2., 2.]),
        >>>                      torch.tensor([3., 3., 3.])],
        >>>                     return_mask=True)
        >>> print(packed)
        tensor([[1., 0., 0.],
                [2., 2., 0.],
                [3., 3., 3.]])
        >>> print(mask)
        tensor([[ True, False, False],
                [ True,  True, False],
                [ True,  True,  True]])

    """
    # If "tensors" is already a Tensor then return it immediately as there is
    # nothing more that can be done. This helps with batch agnostic
    # programming.
    if isinstance(tensors, Tensor):
        return tensors

    # Gather some general setup info
    count, device, dtype = len(tensors), tensors[0].device, tensors[0].dtype

    # Identify the maximum size, if one was not specified.
    if size is None:
        size = torch.tensor([i.shape for i in tensors]).max(0).values


    # Tensor to pack into, filled with padding value.
    padded = torch.full((count, *size), value, dtype=dtype, device=device)

    if return_mask:   # Generate the mask if requested.
        mask = torch.full((count, *size), False, dtype=torch.bool,
                          device=device)

    # Loop over & pack "tensors" into "padded". A proxy index "n" must be used
    # for assignments rather than a slice to prevent in-place errors.
    for n, source in enumerate(tensors):
        # Slice operations not elegant but they are dimension agnostic & fast.
        padded[(n, *[slice(0, s) for s in source.shape])] = source
        if return_mask:  # Update the mask if required.
            mask[(n, *[slice(0, s) for s in source.shape])] = True

    # If "axis" was anything other than 0, then "padded" must be permuted.
    if axis != 0:
        # Resolve relative any axes to their absolute equivalents to maintain
        # expected slicing behaviour when using the insert function.
        axis = padded.dim() + 1 + axis if axis < 0 else axis

        # Build a list of axes indices; but omit the axis on which the data
        # was concatenated (i.e. 0).
        ax = list(range(1, padded.dim()))

        ax.insert(axis, 0)  # Re-insert the concatenation axis as specified

        padded = padded.permute(ax)  # Perform the permeation

        if return_mask:  # Perform permeation on the mask is present.
            mask = mask.permute(ax)

    # Return the packed tensor, and the mask if requested.
    return (padded, mask) if return_mask else padded


def build_coulomb_matrix(geometry: Geometry_, **kwargs):
    """Construct the 1/R matrix for the periodic geometry.

    Arguments:
        geometry: Object for calculation, storing the data of input geometry.

    Keyword Arguments:
        tol_ewald: Ewald tolerance.
        method: Method to obtain parameters of alpha and cutoff.
        nsearchiter: Maximum of iteration for searching alpha, maxg and maxr.

    Examples:
        >>> from tbmalt import Geometry
        >>> from tbmalt.physics.dftb.coulomb import build_coulomb_matrix
        >>> import torch
        >>> cell = torch.tensor([[2., 0., 0.], [0., 4., 0.], [0., 0., 2.]])
        >>> pos = torch.tensor([[0., 0., 0.], [0., 2., 0.]])
        >>> num = torch.tensor([1, 1])
        >>> cutoff = torch.tensor([9.98])
        >>> system = Geometry(num, pos, cell, units='a', cutoff=cutoff)
        >>> invrmat = build_coulomb_matrix(system, method='search')
        >>> print(invrmat)
        tensor([[-0.4778, -0.2729],
                [-0.2729, -0.4778]])

    """
    # Check the type of pbc and choose corresponding subclass
    if geometry.pbc.ndim == 1:  # -> Single
        _sum_dim = geometry.pbc.sum(dim=-1)
    else:  # -> Batch
        _sum_dim = geometry.pbc[0].sum(dim=-1)

    if _sum_dim == 1:  # -> 1D pbc
        coulomb = Ewald1d(geometry, geometry.periodicity, **kwargs)
    elif _sum_dim == 2:  # -> 2D pbc
        coulomb = Ewald2d(geometry, geometry.periodicity, **kwargs)
    elif _sum_dim == 3:  # -> 3D pbc
        coulomb = Ewald3d(geometry, geometry.periodicity, **kwargs)
    else:
        raise ValueError("Number of dimensions should not exceed 3.")

    return coulomb.invrmat


class Ewald(ABC):
    """ABC for calculating the coulombic interaction in periodic geometry.

    `Ewald` class calculates the long range coulombic interaction using Ewald
    summation, which consists of three different parts, i.e. two rapidly
    converging terms (real space sum, reciprocal space sum) and a
    self-correction term. Formulas for different periodic boundary conditions
    are given in the corresponding references.

    Arguments:
        geometry: Object for calculation, storing the data of input geometry.
        periodic: Object for calculation, storing the data of translation
            vectors and neighbour list for periodic boundary condition.
        param: Parameter used for calculation. Cell volume for 3D pbc,
            cell length for 1D & 2D pbc.

    Keyword Arguments:
        tol_ewald: Tolerance of Ewald summation.
        method: Method to obtain parameters of alpha, maxr and maxg.
        nsearchiter: Maximum of iteration for searching alpha, maxg and maxr.

    Attributes:
        geometry: Object for calculation, storing the data of input geometry.
        periodic: Object for calculation, storing the data of translation
            vectors and neighbour list for periodic boundary condition.
        latvec: Lattice vectors of the periodic systems.
        n_atoms: Number of atoms in the system.
        coord : Coordinates of the atoms.
        recvec: Reciprocal lattice vectors.
        param: Parameter used for calculation. Cell volume for 3D pbc,
            cell length for 1D & 2D pbc.
        tol_ewald: Tolerance of Ewald summation.
        method: Method to obtain parameters of alpha, maxr and maxg.
        nsearchiter: Maximum of iteration for searching alpha, maxg and maxr.
        alpha: Splitting parameter for the Ewald summation
        maxr: The longest real space vector that gives a bigger
            contribution to the EWald sum than tolerance.
        maxg: The longest reciprocal vector that gives a bigger
            contribution to the EWald sum than tolerance.
        rcellvec_ud: Cell translation vectors in absolute units.
        ncell_ud: Number of lattice cells.
        distmat: Periodic distances.
        neighbour: A mask to choose atoms of images inside the cutoff.
        ewald_r: Real part of the Ewald summation.
        mask_g: Mask used for calculation of reciprocal part.
        ewald_g: Reciprocal part of the Ewald summation
        invrmat: 1/R matrix for the periodic geometry.

    Notes:
        There are two available methods to generate adjustable parameters for
        Ewald summation. The default method uses empirical formulas from
        experience to obtain alpha, maxr and maxg. While these parameters can
        also be obtained by searching, which give exactly the same results in
        DFTB+ for 3D pbc.

    Warning:
        The result of 1D ewald summation is sensitive to the selection of
        splitting parameter alpha. Using the default method rather than
        searching can achieve the convergence. Besides, mixing of different pbc
        is not supported.

    References:
        [1]: Journal of Computational Physics 285 (2015): 280-315.
        [2]: The Journal of chemical physics 136.16 (2012): 164111.
        [3]: Advances in Computational Mathematics 42.1 (2016): 227-248.
        [4]: Chemical physics letters 340.1-2 (2001): 157-164.

    """

    def __init__(self, geometry: Geometry_, periodic: Periodicity_, param: Tensor,
                 **kwargs):

        # Read input geometry
        self.geometry: Geometry_ = geometry
        self.periodic: Periodicity_ = periodic
        self.latvec: Tensor = self.geometry.lattice
        self.n_atoms: Tensor = self.geometry.n_atoms
        self.coord: Tensor = self.geometry.positions
        self.recvec: Tensor = self.periodic.reciprocal_lattice

        self._device = self.latvec.device
        self._dtype = self.latvec.dtype

        # Number of batches if in batch mode (for internal use only)
        self._n_batch: Optional[int] = (None if self.latvec.dim() == 2
                                        else len(self.latvec))

        # Parameter used for calculation
        self.param: Tensor = param if self._n_batch else param.unsqueeze(0)

        # Tolerance of ewald summation
        self.tol_ewald: Tensor = kwargs.get(
            'tol_ewald', torch.tensor(1e-9, device=self._device,
                                      dtype=self._dtype))

        # Method to obtain parameters for calculation
        self.method: str = kwargs.get('method', 'experience')

        # Maximun number of iteration when searching alpha
        self.nsearchiter: int = kwargs.get('nsearchiter', 30)

        # Maximum number of atoms in geometry
        self._max_natoms: Tensor = torch.max(self.n_atoms)

        # Default method to obtain parameters by empirical formulas
        if self.method == 'experience':

            # Splitting parameter
            self.alpha: Tensor = self._default_alpha()

            ff = torch.sqrt(-torch.log(self.tol_ewald))

            # The longest real space vector
            self.maxr: Tensor = ff / self.alpha

            # The longest reciprocal vector
            self.maxg: Tensor = 2.0 * self.alpha * ff

        else:  # -> Obtain parameters by searching
            self.alpha: Tensor = self._get_alpha()
            self.maxr: Tensor = self._get_maxr()
            self.maxg: Tensor = self._get_maxg()

        # The updated lattice points
        self.rcellvec_ud, self.ncell_ud = self._update_latvec()

        # The updated neighbour lists
        self.distmat, self.neighbour = self._update_neighbour()

        # Real part of the Ewald summation
        self.ewald_r, self.mask_g = self._invr_periodic_real()

        # Reciprocal part of the Ewald summation
        self.ewald_g: Tensor = self._invr_periodic_reciprocal()

        # 1/R matrix for the periodic geometry
        self.invrmat: Tensor = self._invr_periodic()

    @abstractmethod
    def _default_alpha(self) -> Tensor:
        """Returns the default value of alpha."""
        pass

    def _update_latvec(self) -> Tuple[Tensor, Tensor]:
        """Update the lattice points for reciprocal Ewald summation."""
        # TODO: might be best to rename this as it is not really updating the
        #   lattice vectors, but rather the cell translation vectors.

        reciprocal_lattice_vector_inverse = Periodicity_.inverse_lattice_vector(
            self.recvec)

        cell_translation_vector_indices, n_cells = \
            Periodicity_.get_cell_translation_vector_indices(
                reciprocal_lattice_vector_inverse, self.maxg)

        cell_translation_vectors = Periodicity_.get_cell_translation_vectors(
            self.recvec, cell_translation_vector_indices)

        return cell_translation_vectors, n_cells

    def _update_neighbour(self) -> Tuple[Tensor, Tensor]:
        """Update the neighbour lists for real Ewald summation."""

        lattice_vector_inverse = Periodicity_.inverse_lattice_vector(self.latvec)

        cell_translation_vector_indices, n_cells = \
            Periodicity_.get_cell_translation_vector_indices(
                lattice_vector_inverse, self.maxr)

        cell_translation_vectors = Periodicity_.get_cell_translation_vectors(
            self.latvec, cell_translation_vector_indices)

        periodic_distances = Periodicity_.get_periodic_distances(
            cell_translation_vectors, self.geometry.positions,
            self.geometry.n_atoms)

        neighbours = Periodicity_.get_neighbours(periodic_distances, self.maxr)

        return periodic_distances, neighbours

    def _invr_periodic(self) -> Tensor:
        """Calculate the 1/R matrix for the periodic geometry."""
        # Extra contribution for self interaction
        if not self._n_batch:  # -> Single
            extra = torch.eye(self._max_natoms, device=self._device,
                              dtype=self._dtype) * 2.0 * self.alpha / np.sqrt(
                np.pi)
        else:  # -> Batch
            extra = torch.eye(self._max_natoms,
                              device=self._device, dtype=self._dtype
                              ).unsqueeze(0).repeat_interleave(
                self._n_batch, dim=0) * (2.0 * self.alpha / np.sqrt(np.pi)
                                         ).unsqueeze(-1).unsqueeze(-1)

        # TODO: only eqald_r is an issue here
        invr = self.ewald_r + self.ewald_g - extra
        invr[self.mask_g] = 0

        return invr

    def _invr_periodic_reciprocal(self) -> Tensor:
        """Calculate the reciprocal part of 1/R matrix."""
        # Lattice points for the reciprocal sum
        n_low = torch.ceil(torch.clone(self.ncell_ud / 2.0))

        # Single
        if not self._n_batch:
            gvec_tem = self.rcellvec_ud[int(n_low):]
            mask = torch.sum(torch.clone(gvec_tem) ** 2, -1) < self.maxg ** 2
            gvec = gvec_tem[mask]

            # Vectors for calculating the reciprocal Ewald sum
            rr = self.coord.repeat(self.n_atoms, 1) -\
                self.coord.repeat(1, self.n_atoms).view(-1, 3)
            # The reciprocal Ewald sum
            recsum = self._ewald_reciprocal_single(rr, gvec, self.alpha,
                                                   self.param)
            ewald_g = torch.reshape(
                recsum, (self._max_natoms, self._max_natoms))
            ewald_g[self.mask_g] = 0

        # Batch
        # Large values are padded in the end of short vectors
        else:
            gvec_tem = pack([torch.unsqueeze(self.rcellvec_ud[
                ibatch, int(n_low[ibatch]): int(2 * n_low[ibatch] - 1)], 0)
                for ibatch in range(self._n_batch)], value=1e3)
            dd2 = torch.sum(torch.clone(gvec_tem) ** 2, -1)
            mask = dd2 < self.maxg.unsqueeze(-1).unsqueeze(-1) ** 2
            gvec = pack([gvec_tem[ibatch, mask[ibatch]]
                         for ibatch in range(self._n_batch)], value=1e3)

            # Vectors for calculating the reciprocal Ewald sum
            rr = self.coord.repeat(1, self._max_natoms, 1) - self.coord.repeat(
                1, 1, self._max_natoms).view(self._n_batch, -1, 3)

            # The reciprocal Ewald sum
            recsum = self._ewald_reciprocal(rr, gvec, self.alpha, self.param)
            ewald_g = torch.reshape(recsum, (self._n_batch, self._max_natoms,
                                             self._max_natoms))
            ewald_g[self.mask_g] = 0

        return ewald_g

    def _invr_periodic_real(self) -> Tuple[Tensor, Tensor]:
        """Calculate the real part of 1/R matrix."""
        if not self._n_batch:  # -> Single
            ewaldr_tmp = self._ewald_real_single()
        else:  # -> Batch
            ewaldr_tmp = self._ewald_real()

        # Mask for summation
        mask = ewaldr_tmp < float('inf')
        mask_real = self.neighbour & mask
        ewaldr_tmp[~mask_real] = 0
        ewald_r = torch.sum(ewaldr_tmp, dim=-3)

        # Mask used for calculation of reciprocal part
        mask_g = ewald_r == 0

        return ewald_r, mask_g

    def _get_alpha(self) -> Tensor:
        """Get optimal alpha for the Ewald sum from searching."""
        # Mask for zero vector
        maskg = self.recvec.ne(0).any(-1)
        maskr = self.latvec.ne(0).any(-1)

        # Ewald parameter
        if not self._n_batch:  # -> Single
            alphainit = torch.tensor([1.0e-8], device=self._device,
                                     dtype=self._dtype)
            # Length of the shortest vector in reciprocal space
            min_g = torch.sqrt(torch.min(torch.sum(
                self.recvec[maskg] ** 2, -1), 0, keepdim=True).values)

            # Length of the shortest vector in real space
            min_r = torch.sqrt(torch.min(torch.sum(
                self.latvec[maskr] ** 2, -1), 0, keepdim=True).values)

        else:  # -> Batch
            alphainit = torch.tensor([1.0e-8],  device=self._device,
                                     dtype=self._dtype).repeat(self._n_batch)
            # Length of the shortest vector in reciprocal space
            min_g = torch.sqrt(torch.min(torch.sum(
                self.recvec[maskg].unsqueeze(
                    0).view(self._n_batch, -1, 3) ** 2, -1), 1).values)

            # Length of the shortest vector in real space
            min_r = torch.sqrt(torch.min(torch.sum(
                self.latvec[maskr].unsqueeze(
                    0).view(self._n_batch, -1, 3) ** 2, -1), 1).values)

        alpha = torch.clone(alphainit)

        # Difference between reciprocal and real parts of the decrease
        # of Ewald sum.
        diff = self._diff_rec_real(alpha, min_g, min_r, self.param)
        ierror = 0

        # Mask for batch calculation
        mask = diff < - self.tol_ewald

        # Loop to find the alpha
        while (alpha[mask] < float('inf')).all():
            alpha[mask] = 2.0 * alpha[mask]
            diff[mask] = self._diff_rec_real(alpha[mask], min_g[mask],
                                             min_r[mask], self.param[mask])
            mask = diff < - self.tol_ewald
            if (~mask).all():
                break
        if torch.max(alpha >= float('inf')):
            ierror = 1
        elif torch.max(alpha == alphainit):
            ierror = 2

        if ierror == 0:
            alphaleft = 0.5 * alpha
            mask = diff < self.tol_ewald
            while (alpha[mask] < float('inf')).all():
                alpha[mask] = 2.0 * alpha[mask]
                diff[mask] = self._diff_rec_real(alpha[mask], min_g[mask],
                                                 min_r[mask], self.param[mask])
                mask = diff < self.tol_ewald
                if (~mask).all():
                    break

        if torch.max(alpha >= float('inf')):
            ierror = 3

        if ierror == 0:
            alpharight = alpha
            alpha = (alphaleft + alpharight) / 2.0
            iiter = 0
            diff = self._diff_rec_real(alpha, min_g, min_r, self.param)
            mask = torch.abs(diff) > self.tol_ewald
            while iiter <= self.nsearchiter:
                mask_neg = diff < 0
                alphaleft[mask_neg] = alpha[mask_neg]
                alpharight[~mask_neg] = alpha[~mask_neg]
                alpha[mask] = (alphaleft[mask] + alpharight[mask]) / 2.0
                diff[mask] = self._diff_rec_real(alpha[mask], min_g[mask],
                                                 min_r[mask], self.param[mask])
                mask = torch.abs(diff) > self.tol_ewald
                iiter += 1
                if (~mask).all():
                    break
            if iiter > self.nsearchiter:
                ierror = 4

        if ierror != 0:
            raise ValueError('Fail to get optimal alpha for Ewald sum.')

        return alpha

    def _get_maxg(self) -> Tensor:
        """Get the longest reciprocal vector that gives a bigger
        contribution to the Ewald sum than tolerance."""
        ginit = torch.tensor([1.0e-8], device=self._device,
                             dtype=self._dtype).repeat(self.alpha.size(0))
        ierror = 0
        xx = torch.clone(ginit)
        yy = self._gterm(xx, self.alpha, self.param)

        # Mask for batch
        mask = yy > self.tol_ewald

        # Loop
        while (xx[mask] < float('inf')).all():
            xx[mask] = xx[mask] * 2.0
            yy[mask] = self._gterm(
                xx[mask], self.alpha[mask], self.param[mask])
            mask = yy > self.tol_ewald
            if (~mask).all():
                break
        if torch.max(xx >= float('inf')):
            ierror = 1
        elif torch.max(xx == ginit):
            ierror = 2

        if ierror == 0:
            xleft = xx * 0.5
            xright = torch.clone(xx)
            yleft = self._gterm(xleft, self.alpha, self.param)
            yright = torch.clone(yy)
            iiter = 0
            mask = (yleft - yright) > self.tol_ewald

            while iiter <= self.nsearchiter:
                xx[mask] = 0.5 * (xleft[mask] + xright[mask])
                yy[mask] = self._gterm(
                    xx[mask], self.alpha[mask], self.param[mask])
                mask_yy = yy >= self.tol_ewald
                xleft[mask_yy] = xx[mask_yy]
                yleft[mask_yy] = yy[mask_yy]
                xright[~mask_yy] = xx[~mask_yy]
                yright[~mask_yy] = yy[~mask_yy]
                mask = (yleft - yright) > self.tol_ewald
                iiter += 1
                if (~mask).all():
                    break
            if iiter > self.nsearchiter:
                ierror = 3
        if ierror != 0:
            raise ValueError('Fail to get maxg for Ewald sum.')

        return xx

    def _get_maxr(self) -> Tensor:
        """Get the longest real space vector that gives a bigger
           contribution to the Ewald sum than tolerance."""
        rinit = torch.tensor([1.0e-8], device=self._device,
                             dtype=self._dtype).repeat(self.alpha.size(0))
        ierror = 0
        xx = torch.clone(rinit)
        yy = self._rterm(xx, self.alpha)

        # Mask for batch
        mask = yy > self.tol_ewald

        # Loop
        while (xx[mask] < float('inf')).all():
            xx[mask] = xx[mask] * 2.0
            yy[mask] = self._rterm(xx[mask], self.alpha[mask])
            mask = yy > self.tol_ewald
            if (~mask).all():
                break
        if torch.max(xx >= float('inf')):
            ierror = 1
        elif torch.max(xx == rinit):
            ierror = 2

        if ierror == 0:
            xleft = xx * 0.5
            xright = torch.clone(xx)
            yleft = self._rterm(xleft, self.alpha)
            yright = torch.clone(yy)
            iiter = 0
            mask = (yleft - yright) > self.tol_ewald

            while iiter <= self.nsearchiter:
                xx[mask] = 0.5 * (xleft[mask] + xright[mask])
                yy[mask] = self._rterm(xx[mask], self.alpha[mask])
                mask_yy = yy >= self.tol_ewald
                xleft[mask_yy] = xx[mask_yy]
                yleft[mask_yy] = yy[mask_yy]
                xright[~mask_yy] = xx[~mask_yy]
                yright[~mask_yy] = yy[~mask_yy]
                mask = (yleft - yright) > self.tol_ewald
                iiter += 1
                if (~mask).all():
                    break
            if iiter > self.nsearchiter:
                ierror = 3
        if ierror != 0:
            raise ValueError('Fail to get maxg for Ewald sum.')

        return xx

    def _ewald_real(self) -> Tensor:
        """Batch calculation of the Ewald sum in the real part for a certain
        vector length."""
        return torch.erfc(self.alpha.unsqueeze(-1).unsqueeze(
            -1).unsqueeze(-1) * self.distmat) / self.distmat

    def _ewald_real_single(self) -> Tensor:
        """Calculation of the Ewald sum in the real part for a certain
        vector length."""
        return torch.erfc(self.alpha * self.distmat) / self.distmat

    def _diff_rec_real(self, alpha: Tensor, min_g: Tensor,
                       min_r: Tensor, param: Tensor) -> Tensor:
        """Returns the difference between reciprocal and real parts of the
        decrease of Ewald sum."""
        return (self._gterm(4.0 * min_g, alpha, param) - self._gterm(
            5.0 * min_g, alpha, param)) - (self._rterm(2.0 * min_r, alpha) -
                                           self._rterm(3.0 * min_r, alpha))

    @abstractmethod
    def _rterm(self, len_r: Tensor, alpha: Tensor) -> Tensor:
        pass

    @abstractmethod
    def _gterm(self, len_g: Tensor, alpha: Tensor, length: Tensor) -> Tensor:
        pass

    @abstractmethod
    def _ewald_reciprocal_single(self, rr: Tensor, gvec: Tensor,
                                 alpha: Tensor, vol: Tensor) -> Tensor:
        pass

    @abstractmethod
    def _ewald_reciprocal(self, rr: Tensor, gvec: Tensor,
                          alpha: Tensor, vol: Tensor) -> Tensor:
        pass


class Ewald3d(Ewald):
    """Implement of Ewald summation for 3D periodic boundary condition.

    Subclass of the `Ewald` class, containing formulas for 3D pbc.

    Arguments:
        geometry: Object for calculation, storing the data of input geometry.
        periodic: Object for calculation, storing the data of translation
            vectors and neighbour list for periodic boundary condition.

    Attributes:
        invrmat: 1/R matrix for the periodic geometry.

    """

    def __init__(self, geometry: Geometry_, periodic: Periodicity_, **kwargs):
        param = periodic.cellvol
        super().__init__(geometry, periodic, param, **kwargs)

    def _default_alpha(self) -> Tensor:
        """Returns the default value of alpha."""
        return (self.n_atoms / self.param ** 2) ** (1/6) * np.pi ** 0.5

    def _ewald_reciprocal(self, rr: Tensor, gvec: Tensor,
                          alpha: Tensor, vol: Tensor) -> Tensor:
        """Calculate the reciprocal part of the Ewald sum."""
        g2 = torch.sum(gvec ** 2, -1)
        dot = torch.matmul(rr, gvec.transpose(-1, -2))
        recsum = torch.sum((torch.exp(- g2 / (4.0 * alpha.unsqueeze(-1) ** 2))
                            / g2).unsqueeze(-2) * torch.cos(dot), -1)
        tem = 2.0 * recsum * 4.0 * np.pi / vol.unsqueeze(-1)

        return (tem - (np.pi / (self.param * self.alpha ** 2)
                       ).unsqueeze(-1)).unsqueeze(-2)

    def _ewald_reciprocal_single(self, rr: Tensor, gvec: Tensor,
                                 alpha: Tensor, vol: Tensor) -> Tensor:
        """Calculate the reciprocal part of the Ewald sum."""
        g2 = torch.sum(gvec ** 2, -1)
        dot = torch.matmul(rr, gvec.transpose(-1, -2))
        recsum = torch.sum((torch.exp(
            - g2 / (4.0 * alpha ** 2)) / g2) * torch.cos(dot), -1)

        return 2.0 * recsum * 4.0 * np.pi / vol - np.pi / (
            self.param * self.alpha ** 2)

    def _gterm(self, len_g: Tensor, alpha: Tensor, cellvol: Tensor) -> Tensor:
        """Returns the maximum value of the Ewald sum in the
           reciprocal part for a certain vector length."""
        return 4.0 * np.pi * (torch.exp((-0.25 * len_g ** 2) / (alpha ** 2))
                              / (cellvol * len_g ** 2))

    def _rterm(self, len_r: Tensor, alpha: Tensor) -> Tensor:
        """Returns the maximum value of the Ewald sum in the
           real part for a certain vector length."""
        return torch.erfc(alpha * len_r) / len_r


class Ewald2d(Ewald):
    """Implement of Ewald summation for 2D boundary condition.

    Subclass of the `Ewald` class, containing formulas for 2D pbc.

    Arguments:
        geometry: Object for calculation, storing the data of input geometry.
        periodic: Object for calculation, storing the data of translation
            vectors and neighbour list for periodic boundary condition.
        length: The length of each lattice vector.

    Attributes:
        invrmat: 1/R matrix for the periodic geometry.

    """

    def __init__(self, geometry: Geometry_, periodic: Periodicity_, **kwargs):
        self.length: Tensor = geometry.periodicity.get_cell_lengths()

        # Get the minimal length of non-zero terms
        tem = torch.clone(self.length)
        tem[tem.eq(0)] = 1e5
        param = torch.min(tem, -1).values

        super().__init__(geometry, periodic, param, **kwargs)

    def _default_alpha(self) -> Tensor:
        """Returns the default value of alpha."""
        return self.n_atoms ** (1/6) / self.param * np.pi ** 0.5

    def _ewald_reciprocal(self, rr: Tensor, gvec: Tensor,
                          alpha: Tensor, length: Tensor) -> Tensor:
        """Calculate the reciprocal part of the Ewald sum."""

        # Mask of periodic directions
        mask_pd = self.periodic.lattice.ne(0).any(-1)

        # Index to describe non-periodic direction
        index_npd = torch.tensor([0, 1, 2], device=self._device
                                 ).repeat(self._n_batch, 1)[~mask_pd]

        # Lengths of lattice vectors of periodic directions
        length_pd = self.length[mask_pd].reshape(self._n_batch, 2)

        g2 = torch.sum(gvec ** 2, -1)
        gg = torch.sqrt(g2)
        dot = torch.matmul(rr, gvec.transpose(-1, -2))

        # Vectors of the non-periodic direction for reciprocal sum. Different
        # directions can be specified in the batch.
        rr_npe = rr[torch.arange(self._n_batch), :, index_npd]

        # Reciprocal, L
        tem = gg.unsqueeze(-1).transpose(-1, -2) * rr_npe.unsqueeze(-1)

        aa = torch.exp(tem)
        tem2 = gg / (alpha.unsqueeze(-1) * 2.0)

        bb = tem2.unsqueeze(-2) + (alpha.unsqueeze(-1) * rr_npe
                                   ).unsqueeze(-1).repeat_interleave(
                                       tem2.size(-1), -1)

        cc = torch.exp(- tem)
        dd = tem2.unsqueeze(-2) - (alpha.unsqueeze(-1) * rr_npe
                                   ).unsqueeze(-1).repeat_interleave(
                                       tem2.size(-1), -1)

        yyt = aa * torch.erfc(bb) + cc * torch.erfc(dd)
        yy = yyt / gg.unsqueeze(-2)

        # Replace nan values
        yy[yy != yy] = 0
        recl = torch.sum(torch.cos(dot) * yy, -1) * 2.0 * np.pi / (
            length_pd[..., 0] * length_pd[..., 1]).unsqueeze(-1)

        # Reciprocal, 0
        tem3 = torch.exp(- alpha.unsqueeze(-1) ** 2 * rr_npe ** 2
                         ) / alpha.unsqueeze(-1) +\
            (np.pi) ** 0.5 * rr_npe * torch.erf(alpha.unsqueeze(-1) * rr_npe)
        rec0 = tem3 * (- 2.0 * np.pi ** 0.5 / ((
            length_pd[..., 0] * length_pd[..., 1]).unsqueeze(-1)))

        return recl + rec0

    def _ewald_reciprocal_single(self, rr: Tensor, gvec: Tensor, alpha:
                                 Tensor, length: Tensor) -> Tensor:
        """Calculate the reciprocal part of the Ewald sum."""

        # Mask of periodic directions
        mask_pd = self.periodic.lattice.ne(0).any(-1)

        # Index to describe non-periodic direction
        index_npd = torch.tensor([0, 1, 2], device=self._device)[~mask_pd]

        # Lengths of lattice vectors of periodic directions
        length_pd = self.length[mask_pd]

        g2 = torch.sum(gvec ** 2, -1)
        gg = torch.sqrt(g2)
        dot = torch.matmul(rr, gvec.transpose(-1, -2))

        # Reciprocal, L
        tem = gg.unsqueeze(-1).transpose(-1, -2) * rr[..., index_npd]
        aa = torch.exp(tem)
        tem2 = gg / (alpha * 2)
        bb = tem2 + alpha * rr[..., index_npd].repeat_interleave(
            tem2.size(-1), -1)
        cc = torch.exp(- tem)
        dd = tem2 - alpha * rr[..., index_npd].repeat_interleave(
            tem2.size(-1), -1)
        yy = (aa * torch.erfc(bb) + cc * torch.erfc(dd)) / gg
        recl = torch.sum(torch.cos(dot) * yy, -1) * 2.0 * np.pi / (
            length_pd[0] * length_pd[1])

        # Reciprocal, 0
        tem3 = torch.exp(- alpha ** 2 * rr[..., index_npd[0]] ** 2) / alpha + \
            (np.pi) ** 0.5 * rr[..., index_npd[0]] * torch.erf(
                alpha * rr[..., index_npd[0]])
        rec0 = tem3 * (- 2.0 * np.pi ** 0.5 / (length_pd[0] * length_pd[1]))

        return recl + rec0

    def _gterm(self, len_g: Tensor, alpha: Tensor, length: Tensor) -> Tensor:
        """Returns the maximum value of the Ewald sum in the
           reciprocal part for a certain vector length."""
        return (torch.erfc(len_g / (alpha * 2)) * 2
                ) / len_g * np.pi / length ** 2

    def _rterm(self, len_r: Tensor, alpha: Tensor) -> Tensor:
        """Returns the maximum value of the Ewald sum in the
           real part for a certain vector length."""
        return torch.erfc(alpha * len_r) / len_r


class Ewald1d(Ewald):
    """Implement of Ewald summation for 1D boundary condition.

    Subclass of the `Ewald` class, containing formulas for 1D pbc.

    Arguments:
        geometry: Object for calculation, storing the data of input geometry.
        periodic: Object for calculation, storing the data of translation
            vectors and neighbour list for periodic boundary condition.
        length: The length of each lattice vector.

    Attributes:
        invrmat: 1/R matrix for the periodic geometry.

    """

    def __init__(self, geometry: Geometry_, periodic: Periodicity_, **kwargs):
        self.length: Tensor = geometry.periodicity.get_cell_lengths()

        # Get the minimal length of non-zero terms
        tem = torch.clone(self.length)
        tem[tem.eq(0)] = 1e5
        param = torch.min(tem, -1).values

        super().__init__(geometry, periodic, param, **kwargs)

    def _default_alpha(self) -> Tensor:
        """Returns the default value of alpha."""
        return self.n_atoms ** (1/6) / self.param * np.pi ** 0.5

    def _ewald_reciprocal(self, rr: Tensor, gvec: Tensor,
                          alpha: Tensor, length: Tensor) -> Tensor:
        """Calculate the reciprocal part of the Ewald sum."""
        dd = {'dtype': self._dtype, 'device': self._device}

        # Mask of periodic direction
        mask_pd = self.periodic.lattice.ne(0).any(-1)

        # Index of non-periodic directions
        index_npd = torch.tensor([0, 1, 2], device=self._device).repeat(
                self._n_batch, 1)[~mask_pd].reshape(self._n_batch, 2)

        g2 = torch.sum(gvec ** 2, -1)
        dot = torch.matmul(rr, gvec.transpose(-1, -2))

        # Reciprocal, L
        aa = g2 / (4 * alpha.unsqueeze(-1) ** 2)

        bb = (rr[torch.arange(self._n_batch), :, index_npd[:, 0]] ** 2 + rr[
            torch.arange(self._n_batch), :, index_npd[:, 1]]
            ** 2) * alpha.unsqueeze(-1) ** 2

        # Numerical method to calculate integral value
        xx = torch.linspace(10.0 ** -20, 1.0, 5000,  **dd)
        kk0 = torch.tensor([[[torch.trapz(1.0 / xx * torch.exp(
            -iaa / xx - ibb * xx), xx)
                              for iaa in aa[ibatch]]
                             for ibb in bb[ibatch]]
                            for ibatch in range(self._n_batch)],
                            device=self._device, dtype=self._dtype)

        recl = torch.sum(torch.cos(dot) * kk0, -1) * 2.0 / length.unsqueeze(-1)

        # Reciprocal, 0
        rec0 = torch.zeros_like(bb)
        mask = bb != 0

        rec0[mask] = (- euler - torch.log(bb[mask])
                      - special.exp1(bb[mask].cpu()).to(self._device)
                      ) / length.unsqueeze(
                          -1).repeat_interleave(bb.size(-1), -1)[mask]

        return recl + rec0

    def _ewald_reciprocal_single(self, rr: Tensor, gvec: Tensor,
                                 alpha: Tensor, length: Tensor) -> Tensor:
        """Calculate the reciprocal part of the Ewald sum."""
        dd = {'dtype': self._dtype, 'device': self._device}

        # Mask of the periodic direction
        mask_pd = self.periodic.lattice.ne(0).any(-1)

        # Index of non-periodic directions
        index_npd = torch.tensor([0, 1, 2], device=self._device)[~mask_pd]

        g2 = torch.sum(gvec ** 2, -1)
        dot = torch.matmul(rr, gvec.transpose(-1, -2))

        # Reciprocal, L
        aa = g2 / (4 * alpha ** 2)
        bb = (rr[:, index_npd[0]] ** 2 + rr[:, index_npd[1]] ** 2) * alpha ** 2
        xx = torch.linspace(10.0 ** -20, 1.0, 5000, **dd)  # Numerical method
        kk0 = torch.tensor([[torch.trapz(1.0 / xx * torch.exp(
            - iaa / xx - ibb * xx), xx) for iaa in aa] for ibb in bb],
            device=self._device, dtype=self._dtype)
        recl = torch.sum(torch.cos(dot) * kk0, -1) * 2.0 / length

        # Reciprocal, 0
        rec0 = torch.zeros_like(bb)
        mask_bb = bb != 0
        rec0[mask_bb] = (- euler - torch.log(bb[mask_bb])
                         - special.exp1(bb[mask_bb].cpu()).to(self._device
                                                              )) / length

        return recl + rec0

    def _gterm(self, len_g: Tensor, alpha: Tensor, length: Tensor) -> Tensor:
        """Returns the maximum value of the Ewald sum in the
           reciprocal part for a certain vector length."""
        return special.exp1((len_g ** 2 / (4 * alpha ** 2)).cpu()).to(
            self._device) / length

    def _rterm(self, len_r: Tensor, alpha: Tensor) -> Tensor:
        """Returns the maximum value of the Ewald sum in the
           real part for a certain vector length."""
        return torch.erfc(alpha * len_r) / len_r
