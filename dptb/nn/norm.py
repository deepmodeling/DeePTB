import torch
from torch import nn

from e3nn import o3
from e3nn.util.jit import compile_mode
from torch_scatter import scatter_mean

@compile_mode("unsupported")
class TypeNorm(nn.Module):
    """Batch normalization for orthonormal representations

    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.

    Parameters
    ----------
    irreps : `o3.Irreps`
        representation

    eps : float
        avoid division by zero when we normalize by the variance

    momentum : float
        momentum of the running average

    affine : bool
        do we have weight and bias parameters

    reduce : {'mean', 'max'}
        method used to reduce

    """

    def __init__(self, irreps, eps=1e-5, momentum=0.1, affine=True, num_type=1, reduce="mean", normalization="component"):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.num_type = num_type

        num_scalar = sum(mul for mul, ir in self.irreps if ir.is_scalar())
        num_features = self.irreps.num_irreps

        self.register_buffer("running_mean", torch.zeros(num_type, num_scalar))
        self.register_buffer("running_var", torch.ones(num_type, num_features))

        if affine:
            self.weight = nn.Parameter(torch.ones(num_type, num_features))
            self.bias = nn.Parameter(torch.zeros(num_type, num_scalar))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ["mean", "max"], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce

        assert normalization in ["norm", "component"], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps}, momentum={self.momentum})"

    def _roll_avg(self, curr, update):
        mask = (update.norm(dim=-1) > 1e-7)
        out = curr.clone()
        out[mask] = (1 - self.momentum) * curr[mask] + self.momentum * update[mask].detach()
        return out


    def forward(self, input, input_type):
        """evaluate

        Parameters
        ----------
        input : `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        input_type : `torch.Tensor`
            tensor of shape ``(batch)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        """
        
        batch, *size, dim = input.shape
        input = input.reshape(batch, -1, dim)  # [batch, sample, stacked features]

        if self.training:
            new_means = []
            new_vars = []

        fields = []
        ix = 0
        irm = 0
        irv = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:
            d = ir.dim
            field = input[:, :, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d

            # [batch, sample, mul, repr]
            field = field.reshape(batch, -1, mul, d)

            if ir.is_scalar():  # scalars
                if self.training:
                    field_mean = field.mean(1).reshape(batch, mul)  # [batch, mul]
                    field_mean = scatter_mean(field_mean, input_type, dim=0, dim_size=self.num_type)  # [num_type, mul]
                    new_means.append(self._roll_avg(self.running_mean[:, irm : irm + mul], field_mean))
                else:
                    field_mean = self.running_mean[:, irm : irm + mul]
                irm += mul

                # [batch, sample, mul, repr]
                field = field - field_mean.reshape(-1, 1, mul, 1)[input_type]

            if self.training:
                if self.normalization == "norm":
                    field_norm = field.pow(2).sum(3)  # [batch, sample, mul]
                elif self.normalization == "component":
                    field_norm = field.pow(2).mean(3)  # [batch, sample, mul]
                else:
                    raise ValueError("Invalid normalization option {}".format(self.normalization))

                if self.reduce == "mean":
                    field_norm = field_norm.mean(1)  # [batch, mul]
                elif self.reduce == "max":
                    field_norm = field_norm.max(1).values  # [batch, mul]
                else:
                    raise ValueError("Invalid reduce option {}".format(self.reduce))

                field_norm = scatter_mean(field_norm, input_type, dim=0, dim_size=self.num_type)  # [num_type, mul]
                new_vars.append(self._roll_avg(self.running_var[:, irv : irv + mul], field_norm))
            else:
                field_norm = self.running_var[:, irv : irv + mul]
            irv += mul

            field_norm = (field_norm + self.eps).pow(-0.5)  # [(batch,) mul]

            if self.affine:
                weight = self.weight[:, iw : iw + mul]  # [mul]
                iw += mul

                field_norm = field_norm * weight  # [num_type, mul]

            field = field * field_norm.reshape(-1, 1, mul, 1)[input_type]  # [batch, sample, mul, repr]

            if self.affine and ir.is_scalar():  # scalars
                bias = self.bias[:, ib : ib + mul]  # [mul]
                ib += mul
                field += bias.reshape(-1, 1, mul, 1)[input_type]  # [batch, sample, mul, repr]

            fields.append(field.reshape(batch, -1, mul * d))  # [batch, sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        if self.training:
            assert irm == self.running_mean.size(-1)
            assert irv == self.running_var.size(-1)
        if self.affine:
            assert iw == self.weight.size(-1)
            assert ib == self.bias.size(-1)

        if self.training:
            if len(new_means) > 0:
                torch.cat(new_means, dim=-1, out=self.running_mean)
            if len(new_vars) > 0:
                torch.cat(new_vars, dim=-1, out=self.running_var)

        output = torch.cat(fields, dim=2)  # [batch, sample, stacked features]
        return output.reshape(batch, *size, dim)