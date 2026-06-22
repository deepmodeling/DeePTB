from dptb.nn.sktb.hopping import BaseHopping
import torch
from typing import Optional, Dict
from dptb.utils._xitorch.interpolate import Interp1D
from dptb.utils.interpolate.dftbplus_interp import DFTBPlusInterp1D
import logging

log = logging.getLogger(__name__)
class HoppingIntp(BaseHopping):

    def __init__(
            self,
            num_ingrls:int,
            method:str='linear',
            **kwargs,
            ) -> None:
        super().__init__()

        assert method in ['linear', 'cspline'], "Only linear and cspline are supported."
        self.functype = 'dftb'
        self.num_ingrls = num_ingrls
        self.intp_method = method   

    def get_skhij(self, rij, **kwargs):
        
        return self.dftb(rij, **kwargs)
    
    def dftb(self, rij:torch.Tensor, xx:torch.Tensor, yy:torch.Tensor, **kwargs):  
        if not hasattr(self, 'intpfunc'):  #or torch.max(torch.abs(self.xx - xx)) > 1e-5:
            self.xx = xx
            xx = xx.reshape(1, -1).repeat(self.num_ingrls, 1)
            self.intpfunc = Interp1D(xx, method=self.intp_method)
        
        assert yy.shape[0] == self.num_ingrls
        assert len(yy.shape) == 2

        if len(rij.shape) <= 1:
            rij =  torch.tile(rij.reshape([1,-1]), (self.num_ingrls,1))
        elif len(rij.shape) == 2:
            assert rij.shape[0] == self.num_ingrls, "the bond distance shape rij is not correct."
        else:
            raise ValueError("The shape of rij is not correct.")
        # 检查 rij 是否在 xx 的范围内
        min_x, max_x = self.xx.min(), self.xx.max()
        mask_in_range = (rij >= min_x) & (rij <= max_x)
        mask_out_range = ~mask_in_range
        if mask_out_range.any():
            # log.warning("Some rij values are outside the interpolation range and will be set to 0.")
            # 创建 rij 的副本，并将范围外的值替换为范围内的值（例如，使用 min_x）
            rij_modified = rij.clone()
            rij_modified[mask_out_range] = (min_x + max_x) / 2
            yyintp = self.intpfunc(xq=rij_modified, y=yy)
            yyintp[mask_out_range] = 0.0
        else:
            yyintp = self.intpfunc(xq=rij, y=yy)

        return yyintp.T


class HoppingIntpSmooth(BaseHopping):
    """
    Smooth SK integral interpolation using DFTB+ algorithm.

    This class provides smooth interpolation that produces identical results
    to DFTB+ when reading the same SKF files. It replaces the simple linear/cspline
    interpolation of HoppingIntp with a more sophisticated algorithm based on
    the DFTB+ implementation.

    Parameters
    ----------
    num_ingrls : int
        Number of SK integrals (reduced_matrix_element).
    dist_fudge : float, optional
        Extrapolation zone size in Bohr. Default is 1.0 (DFTB+ modern method).
    n_points : int, optional
        Number of points for polynomial interpolation. Default is 8.

    Attributes
    ----------
    functype : str
        Always 'dftb' for this class.
    num_ingrls : int
        Number of SK integral channels.

    Notes
    -----
    Key differences from HoppingIntp:
    1. Uses 8-point polynomial interpolation instead of 2-point linear
    2. Smooth poly5ToZero decay beyond grid instead of hard cutoff
    3. Cutoff is x_max + distFudge instead of just x_max

    This interpolation scheme is based on the DFTB+ implementation for
    compatibility with standard SKF files.

    Examples
    --------
    >>> hopping = HoppingIntpSmooth(num_ingrls=10)
    >>> rij = torch.tensor([1.5, 2.0, 3.5])  # Bond distances
    >>> xx = torch.linspace(0.5, 10.0, 100)   # Distance grid
    >>> yy = torch.randn(10, 100)             # SK integrals
    >>> result = hopping.get_skhij(rij, xx=xx, yy=yy)  # [3, 10]
    """

    def __init__(
        self,
        num_ingrls: int,
        dist_fudge: float = 1.0,
        n_points: int = 8,
        **kwargs,
    ) -> None:
        super().__init__()

        self.functype = 'dftb'
        self.num_ingrls = num_ingrls
        self.dist_fudge = dist_fudge
        self.n_points = n_points

        # Cache for the last used grid
        self._cached_xx: Optional[torch.Tensor] = None

    def get_skhij(self, rij: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Get SK integrals at given bond distances.

        Parameters
        ----------
        rij : torch.Tensor
            Bond distances with shape [n_edges] or [num_ingrls, n_edges].
        **kwargs
            Must include:
            - xx : torch.Tensor - Distance grid [n_grid]
            - yy : torch.Tensor - SK integrals [num_ingrls, n_grid]

        Returns
        -------
        torch.Tensor
            Interpolated SK integrals with shape [n_edges, num_ingrls].
        """
        return self.dftb(rij, **kwargs)

    def dftb(
        self,
        rij: torch.Tensor,
        xx: torch.Tensor,
        yy: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Interpolate SK integrals using DFTB+ algorithm.

        Parameters
        ----------
        rij : torch.Tensor
            Bond distances [n_edges] or [num_ingrls, n_edges].
        xx : torch.Tensor
            Distance grid [n_grid].
        yy : torch.Tensor
            SK integrals [num_ingrls, n_grid].

        Returns
        -------
        torch.Tensor
            Interpolated values [n_edges, num_ingrls].
        """
        # Create or retrieve cached interpolator
        if not self._is_same_grid(xx):
            self._create_interpolator(xx)

        # Validate yy shape
        assert yy.shape[0] == self.num_ingrls, \
            f"Expected {self.num_ingrls} integrals, got {yy.shape[0]}"
        assert len(yy.shape) == 2, \
            f"Expected 2D tensor for yy, got {len(yy.shape)}D"

        # Handle rij shape
        if len(rij.shape) <= 1:
            # Simple 1D case: [n_edges]
            rij_1d = rij.flatten()
        elif len(rij.shape) == 2:
            assert rij.shape[0] == self.num_ingrls, \
                f"Expected rij shape [num_ingrls, n_edges], got {rij.shape}"
            if not torch.allclose(rij, rij[0].expand_as(rij)):
                raise ValueError("Expected all rows of 2D rij to contain the same bond distances.")
            # Take first row (all rows should be same for uniform query)
            rij_1d = rij[0]
        else:
            raise ValueError(f"Invalid rij shape: {rij.shape}")

        # Interpolate using DFTB+ method
        # DFTBPlusInterp1D handles all regions internally
        yyintp = self._interp(rij_1d, yy)

        return yyintp.T  # [n_edges, num_ingrls]

    def _is_same_grid(self, xx: torch.Tensor) -> bool:
        """Check if the grid is the same as cached."""
        if self._cached_xx is None:
            return False
        if self._cached_xx.shape != xx.shape:
            return False
        return torch.allclose(self._cached_xx, xx, atol=1e-10)

    def _create_interpolator(self, xx: torch.Tensor) -> None:
        """Create and cache a new interpolator for the given grid."""
        self._cached_xx = xx.clone()
        self._interpolator = DFTBPlusInterp1D(
            x=xx,
            y=None,  # y will be provided at call time
            dist_fudge=self.dist_fudge,
            n_points=self.n_points,
        )
        log.debug(
            f"Created DFTB+ interpolator: grid [{xx.min():.3f}, {xx.max():.3f}] Å, "
            f"cutoff {self._interpolator.x_cutoff:.3f} Å"
        )

    def _interp(self, rij: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        """
        Perform interpolation.

        Parameters
        ----------
        rij : torch.Tensor
            Query points [n_edges].
        yy : torch.Tensor
            SK integrals [num_ingrls, n_grid].

        Returns
        -------
        torch.Tensor
            Interpolated values [num_ingrls, n_edges].
        """
        return self._interpolator(rij, yy)

    def get_cutoff(self) -> float:
        """
        Get the effective cutoff distance.

        Returns
        -------
        float
            Cutoff distance in Angstrom, or 0 if no grid has been set.
        """
        if hasattr(self, '_interpolator'):
            return self._interpolator.get_cutoff()
        return 0.0