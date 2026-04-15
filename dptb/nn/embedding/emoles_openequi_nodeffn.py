from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import nn
from e3nn import o3
from e3nn.o3 import Linear

from dptb.nn.embedding.emb import Embedding
from dptb.nn.norm import SeperableLayerNorm

from .emoles import EMolES, EAMPOpenequi, OEQTensorProduct, create_gate, oeq


def scale_irreps_mul(irreps: o3.Irreps, factor: float) -> o3.Irreps:
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    scaled = []
    for mul, ir in o3.Irreps(irreps):
        scaled_mul = max(1, int(round(mul * factor)))
        scaled.append((scaled_mul, ir))
    return o3.Irreps(scaled).simplify()


class EquivariantNodeFFN(nn.Module):
    def __init__(
        self,
        irreps: o3.Irreps,
        hidden_factor: float = 4.0,
        norm_eps: float = 1e-8,
        ln_flag: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        hidden_irreps = scale_irreps_mul(self.irreps, hidden_factor)

        if ln_flag:
            self.norm = SeperableLayerNorm(
                irreps=self.irreps,
                eps=norm_eps,
                affine=True,
                normalization="component",
                std_balance_degrees=True,
                dtype=dtype,
                device=device,
            )
        else:
            self.norm = nn.Identity()

        self.activation = create_gate(hidden_irreps)
        self.lin1 = Linear(
            self.irreps,
            self.activation.irreps_in,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )
        self.lin2 = Linear(
            self.activation.irreps_out,
            self.irreps,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )
        self.alpha = nn.Parameter(torch.tensor(0.0, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = self.lin2(self.activation(self.lin1(self.norm(x))))
        return x + torch.tanh(self.alpha) * dx


class EAMPOpenequiNodeFFN(EAMPOpenequi):
    def __init__(self, **kwargs):
        layer_kwargs = dict(kwargs)
        self.ffn_hidden_factor = float(layer_kwargs.pop("ffn_hidden_factor", 0.0))
        self.use_node_ffn = bool(layer_kwargs.pop("use_node_ffn", False))
        super().__init__(**layer_kwargs)

        self.node_ffn = None
        if self.use_node_ffn and self.ffn_hidden_factor > 1.0:
            self.node_ffn = EquivariantNodeFFN(
                self.irreps_out,
                hidden_factor=self.ffn_hidden_factor,
                norm_eps=self.norm_eps,
                ln_flag=layer_kwargs.get("ln_flag", True),
                dtype=self.dtype,
                device=self.device,
            )

    def forward(
        self,
        latents: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        atom_type: torch.Tensor,
        node_onehot: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vector: torch.Tensor,
        active_edges: torch.Tensor,
        wigner_D_all: torch.Tensor | None,
    ):
        node_features, edge_features, wigner_D_all = super().forward(
            latents,
            node_features,
            edge_features,
            atom_type,
            node_onehot,
            edge_index,
            edge_vector,
            active_edges,
            wigner_D_all,
        )
        if self.node_ffn is not None:
            node_features = self.node_ffn(node_features)
        return node_features, edge_features, wigner_D_all


def _create_nodeffn_layer_worker(args):
    idx, layer_kwargs = args
    t_start = time.time()
    layer = EAMPOpenequiNodeFFN(**layer_kwargs)
    duration = time.time() - t_start
    return idx, layer, duration


def _create_tp_worker(args):
    name, tp_kwargs = args
    t_start = time.time()
    tp = OEQTensorProduct(**tp_kwargs)
    duration = time.time() - t_start
    return name, tp, duration


@Embedding.register("emoles_openequi_nodeffn")
class EMolESOpenequiNodeFFN(EMolES):
    """
    EMolES OpenEqui ablation with the original separable norm and Gate path,
    plus an optional node-wise equivariant FFN after each hidden layer.
    """

    def __init__(self, **kwargs):
        n_layers = kwargs.get("n_layers", 3)
        irreps_hidden = kwargs.get("irreps_hidden")
        use_interpolation_out = kwargs.get("use_interpolation_out", True)
        edge_one_hot_dim = kwargs.get("edge_one_hot_dim", 128)
        ffn_hidden_factor = float(kwargs.get("ffn_hidden_factor", 0.0))
        ffn_apply_to_last = bool(kwargs.get("ffn_apply_to_last", False))

        super().__init__(**kwargs)

        if oeq is None:
            raise ImportError("OpenEquivariance is not installed.")

        self.layers = torch.nn.ModuleList([None] * n_layers)
        irreps_hidden_obj = o3.Irreps(irreps_hidden)
        orbpair_irreps = self.idp.orbpair_irreps.sort()[0].simplify()

        base_layer_kwargs = {
            "latent_dim": kwargs.get("latent_dim", 128),
            "norm_eps": kwargs.get("norm_eps", 1e-8),
            "radial_emb": kwargs.get("tp_radial_emb", False),
            "radial_channels": kwargs.get("tp_radial_channels", [128, 128]),
            "res_update": kwargs.get("res_update", True),
            "use_layer_onehot_tp": kwargs.get("use_layer_onehot_tp", True),
            "res_update_ratios": kwargs.get("res_update_ratios", None),
            "res_update_ratios_learnable": kwargs.get("res_update_ratios_learnable", False),
            "avg_num_neighbors": kwargs.get("avg_num_neighbors", None),
            "dtype": self.dtype,
            "device": self.device,
            "ln_flag": kwargs.get("ln_flag", True),
            "in_frame_flag": kwargs.get("in_frame_flag", True),
            "optimized_in_frame": kwargs.get("optimized_in_frame", True),
            "onehot_mode": kwargs.get("onehot_mode", "FullTP"),
            "self_mix_flag": kwargs.get("self_mix_flag", False),
            "self_mix_mode": kwargs.get("self_mix_mode", "scalar_channelwise"),
            "self_mix_iter": kwargs.get("self_mix_iter", 1),
            "self_mix_type": kwargs.get("self_mix_type", "node"),
            "ffn_hidden_factor": ffn_hidden_factor,
        }

        tasks = []
        for i in range(n_layers):
            if i == 0:
                irreps_in_layer = self.init_layer.irreps_out
            else:
                irreps_in_layer = irreps_hidden_obj

            if self.in_frame_flag:
                rotate_in = (i == 0)
                rotate_out = (i == n_layers - 1)
            else:
                rotate_in, rotate_out = True, True

            if i == n_layers - 1:
                irreps_out_layer = orbpair_irreps
                use_interpolation_tp = bool(use_interpolation_out)
            else:
                irreps_out_layer = irreps_hidden_obj
                use_interpolation_tp = False

            use_node_ffn = ffn_hidden_factor > 1.0 and ((i < n_layers - 1) or ffn_apply_to_last)

            current_kwargs = base_layer_kwargs.copy()
            current_kwargs.update(
                {
                    "node_irreps_in": irreps_in_layer,
                    "edge_irreps_in": irreps_in_layer,
                    "irreps_out": irreps_out_layer,
                    "tp_rotate_in": rotate_in,
                    "tp_rotate_out": rotate_out,
                    "use_interpolation_tp": use_interpolation_tp,
                    "node_one_hot_dim": self.n_atom,
                    "use_node_ffn": use_node_ffn,
                }
            )
            tasks.append((i, current_kwargs))

        print(f"Starting parallel compilation for {n_layers} node-FFN layers...")
        t_start_all = time.time()

        with ThreadPoolExecutor(max_workers=8) as executor:
            layer_futures = [executor.submit(_create_nodeffn_layer_worker, task) for task in tasks]

            tp_futures = []
            if self.use_out_onehot_tp:
                tp1_kwargs = {
                    "irreps_in1": self.node_irreps_out,
                    "irreps_in2": o3.Irreps(f"{self.n_atom}x0e"),
                    "irreps_out": self.idp.orbpair_irreps,
                    "tp_mode": "uvw",
                }
                tp2_kwargs = {
                    "irreps_in1": self.edge_irreps_out,
                    "irreps_in2": o3.Irreps(f"{edge_one_hot_dim}x0e"),
                    "irreps_out": self.idp.orbpair_irreps,
                    "tp_mode": "uvw",
                }
                tp_futures.append(executor.submit(_create_tp_worker, ("out_node_ele_tp", tp1_kwargs)))
                tp_futures.append(executor.submit(_create_tp_worker, ("out_edge_ele_tp", tp2_kwargs)))

            for future in layer_futures:
                idx, layer, _ = future.result()
                self.layers[idx] = layer

            for future in tp_futures:
                name, tp_module, _ = future.result()
                setattr(self, name, tp_module)

        print(f"Node-FFN compilation finished in {time.time() - t_start_all:.2f}s")
