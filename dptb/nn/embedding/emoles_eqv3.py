from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
from torch import nn
from e3nn import o3
from e3nn.o3 import Linear

from dptb.nn.embedding.emb import Embedding
from dptb.nn.tensor_product import SO2_Linear

from .emoles import EMolES, EAMPOpenequi, OEQTensorProduct, oeq
from .lem_moe_v3_plugins import (
    FlatSwiGLUS2Merge,
    build_equivariant_norm,
    build_gate_activation,
    can_use_flat_s2_patch,
)


def _create_eqv3_layer_worker(args):
    idx, layer_kwargs = args
    t_start = time.time()
    layer = EAMPOpenequiEqV3(**layer_kwargs)
    duration = time.time() - t_start
    return idx, layer, duration


def _create_tp_worker(args):
    name, tp_kwargs = args
    t_start = time.time()
    tp = OEQTensorProduct(**tp_kwargs)
    duration = time.time() - t_start
    return name, tp, duration


class EAMPOpenequiEqV3(EAMPOpenequi):
    def __init__(self, **kwargs):
        layer_kwargs = dict(kwargs)
        self.ln_flag = layer_kwargs.get("ln_flag", True)
        self.equivariant_norm_type = layer_kwargs.pop("equivariant_norm_type", "merged_rms")
        self.hidden_edge_activation_type = layer_kwargs.pop("hidden_edge_activation_type", "swiglu_s2")
        self.swiglu_s2_grid_resolution = tuple(layer_kwargs.pop("swiglu_s2_grid_resolution", [14, 14]))
        super().__init__(**layer_kwargs)

        self.sln_n = self._build_eq_norm(self.node_irreps_in)
        self.sln_e = self._build_eq_norm(self.edge_irreps_in)
        self.activation = self._build_main_activation(self.hidden_edge_activation_type)

        real_tp_rotate_out = self.tp_rotate_out
        if self.in_frame_flag and self.optimized_in_frame:
            real_tp_rotate_out = False

        tp_irreps_out = getattr(self.activation, "tp_main_irreps", self.activation.irreps_in)
        extra_m0_outsize = getattr(self.activation, "extra_m0_outsize", 0)
        self.tp = SO2_Linear(
            irreps_in=self.node_irreps_in + self.edge_irreps_in + self.node_irreps_in,
            irreps_out=tp_irreps_out,
            latent_dim=kwargs.get("latent_dim", 128),
            radial_emb=kwargs.get("radial_emb", False),
            radial_channels=kwargs.get("radial_channels", [128, 128]),
            extra_m0_outsize=extra_m0_outsize,
            use_interpolation=kwargs.get("use_interpolation_tp", False),
            rotate_in=self.tp_rotate_in,
            rotate_out=real_tp_rotate_out,
        )
        self.lin_post = Linear(
            self.activation.irreps_out,
            self.irreps_out,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )

    def _build_eq_norm(self, irreps: o3.Irreps) -> nn.Module:
        if not self.ln_flag:
            return nn.Identity()
        norm = build_equivariant_norm(
            self.equivariant_norm_type,
            irreps,
            self.norm_eps,
            self.dtype,
            self.device,
        )
        if norm is None:
            return nn.Identity()
        return norm

    def _build_main_activation(self, activation_type: str) -> nn.Module:
        if activation_type == "gate":
            return build_gate_activation(self.irreps_out)
        if activation_type == "swiglu_s2":
            if not can_use_flat_s2_patch(self.irreps_out):
                raise ValueError(
                    f"SwiGLU-S2 requires uniform hidden irreps, got {self.irreps_out}."
                )
            return FlatSwiGLUS2Merge(
                self.irreps_out,
                grid_resolution=self.swiglu_s2_grid_resolution,
            )
        raise ValueError(f"Unsupported hidden_edge_activation_type={activation_type!r}")

    def _build_mixer_module(self):
        mixer = torch.nn.ModuleDict()
        mixer["norm"] = self._build_eq_norm(self.irreps_out)

        l0_indices = self.l0_indices
        scalar_dim = len(l0_indices)

        gate = build_gate_activation(self.irreps_out)
        mixer["gate"] = gate

        tps = nn.ModuleList()
        pre_gate_linear = None

        for _ in range(self.self_mix_iter):
            tp_layer = None

            if "scalar" in self.self_mix_mode:
                irreps_in2 = o3.Irreps(f"{scalar_dim}x0e")
                if "full" in self.self_mix_mode:
                    tp_layer = OEQTensorProduct(
                        irreps_in1=self.irreps_out,
                        irreps_in2=irreps_in2,
                        irreps_out=gate.irreps_in,
                        tp_mode="uvw",
                    )
                elif "channelwise" in self.self_mix_mode:
                    tp_layer = OEQTensorProduct(
                        irreps_in1=self.irreps_out,
                        irreps_in2=irreps_in2,
                        irreps_out=self.irreps_out,
                        tp_mode="uvu",
                    )
                    if pre_gate_linear is None:
                        pre_gate_linear = Linear(
                            self.irreps_out,
                            gate.irreps_in,
                            internal_weights=True,
                            shared_weights=True,
                        )
                else:
                    raise ValueError(f"Unknown scalar mode: {self.self_mix_mode}")
            elif "full_full" in self.self_mix_mode:
                irreps_in2 = self.irreps_out
                if "uvu" in self.self_mix_mode:
                    tp_layer = OEQTensorProduct(
                        irreps_in1=self.irreps_out,
                        irreps_in2=irreps_in2,
                        irreps_out=self.irreps_out,
                        tp_mode="uvu",
                    )
                    if pre_gate_linear is None:
                        pre_gate_linear = Linear(
                            self.irreps_out,
                            gate.irreps_in,
                            internal_weights=True,
                            shared_weights=True,
                        )
                elif "uuw" in self.self_mix_mode:
                    tp_layer = OEQTensorProduct(
                        irreps_in1=self.irreps_out,
                        irreps_in2=irreps_in2,
                        irreps_out=self.irreps_out,
                        tp_mode="uuw",
                    )
                    if pre_gate_linear is None:
                        pre_gate_linear = Linear(
                            self.irreps_out,
                            gate.irreps_in,
                            internal_weights=True,
                            shared_weights=True,
                        )
                else:
                    raise ValueError(f"Unknown full_full mode: {self.self_mix_mode}")
            else:
                raise ValueError(f"Unknown self_mix_mode: {self.self_mix_mode}")

            tps.append(tp_layer)

        mixer["tps"] = tps
        if pre_gate_linear is not None:
            mixer["pre_gate_linear"] = pre_gate_linear

        mixer["post_linear"] = Linear(
            gate.irreps_out,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )
        return mixer


@Embedding.register("emoles_openequi_eqv3")
class EMolESOpenequiEqV3(EMolES):
    """
    EMolES with OpenEquivariance tensor products plus EqV3-style merged RMS norm
    and optional flat SwiGLU-S2 activation on hidden layers.
    """

    def __init__(self, **kwargs):
        n_layers = kwargs.get("n_layers", 3)
        irreps_hidden = kwargs.get("irreps_hidden")
        use_interpolation_out = kwargs.get("use_interpolation_out", True)
        edge_one_hot_dim = kwargs.get("edge_one_hot_dim", 128)
        ln_flag = kwargs.get("ln_flag", True)
        equivariant_norm_type = kwargs.get("equivariant_norm_type", "merged_rms")
        hidden_edge_activation_type = kwargs.get("hidden_edge_activation_type", "swiglu_s2")
        swiglu_s2_grid_resolution = kwargs.get("swiglu_s2_grid_resolution", [14, 14])

        super().__init__(**kwargs)

        if oeq is None:
            raise ImportError("OpenEquivariance is not installed.")

        if ln_flag:
            init_norm = build_equivariant_norm(
                equivariant_norm_type,
                self.init_layer.irreps_out,
                kwargs.get("norm_eps", 1e-8),
                self.dtype,
                self.device,
            )
            if init_norm is None:
                self.init_layer.sln_n = nn.Identity()
            else:
                self.init_layer.sln_n = init_norm
        else:
            self.init_layer.sln_n = nn.Identity()

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
            "ln_flag": ln_flag,
            "in_frame_flag": kwargs.get("in_frame_flag", True),
            "optimized_in_frame": kwargs.get("optimized_in_frame", True),
            "onehot_mode": kwargs.get("onehot_mode", "FullTP"),
            "self_mix_flag": kwargs.get("self_mix_flag", False),
            "self_mix_mode": kwargs.get("self_mix_mode", "scalar_channelwise"),
            "self_mix_iter": kwargs.get("self_mix_iter", 1),
            "self_mix_type": kwargs.get("self_mix_type", "node"),
            "equivariant_norm_type": equivariant_norm_type,
            "swiglu_s2_grid_resolution": swiglu_s2_grid_resolution,
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
                activation_type = "gate"
            else:
                irreps_out_layer = irreps_hidden_obj
                use_interpolation_tp = False
                activation_type = hidden_edge_activation_type

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
                    "hidden_edge_activation_type": activation_type,
                }
            )
            tasks.append((i, current_kwargs))

        print(f"Starting parallel compilation for {n_layers} EqV3-style layers...")
        t_start_all = time.time()

        with ThreadPoolExecutor(max_workers=8) as executor:
            layer_futures = [executor.submit(_create_eqv3_layer_worker, task) for task in tasks]

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

        print(f"EqV3-style compilation finished in {time.time() - t_start_all:.2f}s")
