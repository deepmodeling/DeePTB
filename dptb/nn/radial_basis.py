from typing import Optional
import math

import torch

from torch import nn

from e3nn.math import soft_one_hot_linspace
from e3nn.util.jit import compile_mode


@compile_mode("trace")
class e3nn_basis(nn.Module):
    r_max: float
    r_min: float
    e3nn_basis_name: str
    num_basis: int

    def __init__(
        self,
        r_max: float,
        r_min: Optional[float] = None,
        e3nn_basis_name: str = "gaussian",
        num_basis: int = 8,
    ):
        super().__init__()
        self.r_max = r_max
        self.r_min = r_min if r_min is not None else 0.0
        self.e3nn_basis_name = e3nn_basis_name
        self.num_basis = num_basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return soft_one_hot_linspace(
            x,
            start=self.r_min,
            end=self.r_max,
            number=self.num_basis,
            basis=self.e3nn_basis_name,
            cutoff=True,
        )

    def _make_tracing_inputs(self, n: int):
        return [{"forward": (torch.randn(5, 1),)} for _ in range(n)]


class BesselBasis(nn.Module):
    r_max: float
    prefactor: float

    def __init__(self, r_max, num_basis=8, trainable=True):
        r"""Radial Bessel Basis, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)

        return self.prefactor * (numerator / x.unsqueeze(-1))


def gaussian_smearing(distances, offset, widths, centered=False):
    if not centered:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components
        diff = distances[..., None] - offset
    else:
        # if Gaussian functions are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances[..., None]
    # compute smear distance values
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    return gauss


class GaussianBasis(nn.Module):
    def __init__(
            self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False
    ):
        super(GaussianBasis, self).__init__()
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.Tensor((offset[1] - offset[0]) * torch.ones_like(offset)) # FloatTensor
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered

    def forward(self, distances):
        """Compute smeared-gaussian distance values.

        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_g) shape.

        """
        return gaussian_smearing(
            distances, self.offsets, self.width, centered=self.centered
        )
