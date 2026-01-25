"""
This file refactor the SK and E3 Rotation in dptb/hamiltonian/transform_se3.py], it will take input of AtomicDataDict.Type
perform rotation from irreducible matrix element / sk parameters in EDGE/NODE FEATURE, and output the atomwise/ pairwise hamiltonian
as the new EDGE/NODE FEATURE. The rotation should also be a GNN module and speed uptable by JIT. The HR2HK should also be included here.
The indexmapping should ne passed here.
"""

import torch
from e3nn.o3 import wigner_3j, Irrep, xyz_to_angles, Irrep
from dptb.utils.constants import h_all_types, anglrMId
from typing import Tuple, Union, Dict
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict
import re
from torch_runstats.scatter import scatter
from dptb.nn.tensor_product import wigner_D
from dptb.nn.sktb.socbasic import get_soc_matrix_cubic_basis
from dptb.utils.tools import float2comlex
#TODO: 1. jit acceleration 2. GPU support 3. rotate AB and BA bond together.

# The `E3Hamiltonian` class is a PyTorch module that represents a Hamiltonian for a system with a
# given basis and can perform forward computations on input data.

import torch.nn as nn
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict


from typing import Dict, Union, Optional


class E3Hamiltonian(torch.nn.Module):
    def __init__(
        self,
        basis: Dict[str, Union[str, list]] = None,
        idp: Union["OrbitalMapper", None] = None,
        decompose: bool = False,
        edge_field: str = "edge_features",   # AtomicDataDict.EDGE_FEATURES_KEY
        node_field: str = "node_features",   # AtomicDataDict.NODE_FEATURES_KEY
        overlap: bool = False,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        rotation: bool = False,
        soc: bool = False,  # 注意：这是“显式 SOC onsite 参数”那条老逻辑，不等同于 has_soc
        # ---- Debug flags ----
        debug: bool = False,
        debug_every: int = 1,
        debug_max_pairtypes: int = 12,
        debug_sample_rows: int = 1,
        **kwargs,
    ) -> None:
        super(E3Hamiltonian, self).__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if isinstance(device, str):
            device = torch.device(device)

        self.overlap = overlap
        self.dtype = dtype
        self.device = device
        self.soc = soc

        self.decompose = decompose
        self.rotation = rotation
        if rotation:
            assert self.decompose, "Rotation is only supported when decompose is True."

        self.edge_field = edge_field
        self.node_field = node_field

        # ---- Debug state ----
        self.debug = bool(debug)
        self.debug_every = int(debug_every) if debug_every is not None else 1
        self.debug_max_pairtypes = int(debug_max_pairtypes)
        self.debug_sample_rows = int(debug_sample_rows)
        self._forward_calls = 0

        # ---- OrbitalMapper ----
        if basis is not None:
            # 如果你确实想支持 basis->idp 直接构建，请把 has_soc/soc_complex_doubling 传进去
            has_soc = bool(kwargs.get("has_soc", False))
            soc_complex_doubling = bool(kwargs.get("soc_complex_doubling", True))
            self.idp = OrbitalMapper(
                basis, method="e3tb", device=self.device,
                has_soc=has_soc, soc_complex_doubling=soc_complex_doubling
            )
            if idp is not None:
                # 建议你扩展 OrbitalMapper.__eq__，至少比较 soc_complex_doubling 和 chemical_symbol_to_type
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        self.basis = self.idp.basis

        # ---- CG basis ----
        self.cgbasis = {}
        self.idp.get_orbpairtype_maps()
        for orbpair in self.idp.orbpairtype_maps.keys():
            self._initialize_CG_basis(orbpair)

        # ---- Optional explicit SOC onsite param (legacy path) ----
        if self.soc:
            if not hasattr(self.idp, "get_orbpair_soc_maps"):
                raise AttributeError("OrbitalMapper has no get_orbpair_soc_maps(); cannot enable soc in E3Hamiltonian.")
            self.idp.get_orbpair_soc_maps()

            self.soc_base_matrix = {
                's': get_soc_matrix_cubic_basis(orbital='s', device=self.device, dtype=self.dtype),
                'p': get_soc_matrix_cubic_basis(orbital='p', device=self.device, dtype=self.dtype),
                'd': get_soc_matrix_cubic_basis(orbital='d', device=self.device, dtype=self.dtype),
            }
            self.cdtype = float2comlex(self.dtype)

    def _infer_n_node(self, data) -> int:
        if self.node_field in data:
            return data[self.node_field].shape[0]
        if "pos" in data:
            return data["pos"].shape[0]
        raise KeyError("Cannot infer n_node: missing node_field and pos in data.")

    # ---------------- Debug helpers ----------------

    def _soc_factor(self) -> int:
        """SOC factor for chunk counting: 4 spin blocks * (2 if Re/Im doubling else 1)."""
        has_soc = bool(getattr(self.idp, "has_soc", False))
        if not has_soc:
            return 1
        doubling = bool(getattr(self.idp, "soc_complex_doubling", True))
        return 4 * (2 if doubling else 1)

    def _soc_channel_names(self):
        has_soc = bool(getattr(self.idp, "has_soc", False))
        if not has_soc:
            return ["(no-soc)"]
        doubling = bool(getattr(self.idp, "soc_complex_doubling", True))
        if doubling:
            return ["UU_re", "UD_re", "DU_re", "DD_re", "UU_im", "UD_im", "DU_im", "DD_im"]
        else:
            return ["UU", "UD", "DU", "DD"]

    def _debug_print_pairtype(self, tag: str, opairtype: str, width: int, n_rme: int):
        has_soc = bool(getattr(self.idp, "has_soc", False))
        factor = self._soc_factor()
        if width % n_rme != 0:
            raise ValueError(
                f"[E3Hamiltonian Debug] {tag} {opairtype}: width={width} not divisible by n_rme={n_rme}. "
                f"This means your feature packing is not chunked by (2l1+1)*(2l2+1)."
            )
        n_chunk = width // n_rme
        msg = (f"[E3Ham-Debug:{tag}] {opairtype}  width={width}  n_rme={n_rme}  n_chunk={n_chunk}")
        if has_soc:
            if n_chunk % factor != 0:
                raise ValueError(
                    f"[E3Hamiltonian Debug] {tag} {opairtype}: n_chunk={n_chunk} not divisible by factor={factor}. "
                    f"Expected chunk layout = n_pair * factor, factor={factor} from SOC blocks."
                )
            n_pair_est = n_chunk // factor
            msg += f"  has_soc=True  factor={factor}  n_pair_est={n_pair_est}  channels={self._soc_channel_names()}"
        else:
            msg += "  has_soc=False"
        print(msg)

    def _debug_chunk_norms(self, tag: str, tensor_2d: torch.Tensor, sli: slice, n_rme: int, n_show: int = 8):
        """
        tensor_2d: (N, D). We take first rows and show chunk norms.
        """
        if tensor_2d.numel() == 0:
            return
        width = sli.stop - sli.start
        n_chunk = width // n_rme
        rows = min(self.debug_sample_rows, tensor_2d.shape[0])
        x = tensor_2d[:rows, sli]  # (rows, width)
        x = x.reshape(rows, n_chunk, n_rme)
        # Frobenius norm per chunk
        norms = torch.linalg.vector_norm(x, dim=-1)  # (rows, n_chunk)
        norms = norms[:, :min(n_show, n_chunk)].detach().cpu()
        print(f"[E3Ham-Debug:{tag}] chunk_norms(first {rows} rows, first {min(n_show,n_chunk)} chunks):\n{norms}")

    # ---------------- Legacy SOC onsite param ----------------

    def _apply_soc(self, data, n_node: int):
        if not self.soc:
            return data
        # 原样保留你的实现；这里只在 debug 时多打印
        if self.debug:
            print("[E3Ham-Debug] _apply_soc enabled (legacy SOC-param path).")
        # --- paste your existing _apply_soc logic here unchanged ---
        # (略：你可以直接用你原来的 _apply_soc)
        return data

    # ---------------- Forward ----------------

    def forward(self, data):
        self._forward_calls += 1
        do_debug = self.debug and (self._forward_calls % max(1, self.debug_every) == 0)

        # 基本一致性检查（不改动你的原逻辑）
        assert data[self.edge_field].shape[1] == self.idp.reduced_matrix_element, \
            f"edge_field width {data[self.edge_field].shape[1]} != idp.rme {self.idp.reduced_matrix_element}"
        if not self.overlap:
            assert data[self.node_field].shape[1] == self.idp.reduced_matrix_element, \
                f"node_field width {data[self.node_field].shape[1]} != idp.rme {self.idp.reduced_matrix_element}"

        n_edge = data["edge_index"].shape[1]  # AtomicDataDict.EDGE_INDEX_KEY
        n_node = self._infer_n_node(data)

        if do_debug:
            print("\n" + "=" * 80)
            print(f"[E3Ham-Debug] forward_call={self._forward_calls}  decompose={self.decompose}  rotation={self.rotation}")
            print(f"[E3Ham-Debug] edge_tensor: shape={tuple(data[self.edge_field].shape)} dtype={data[self.edge_field].dtype}")
            if not self.overlap:
                print(f"[E3Ham-Debug] node_tensor: shape={tuple(data[self.node_field].shape)} dtype={data[self.node_field].dtype}")
            print(f"[E3Ham-Debug] idp.has_soc={bool(getattr(self.idp,'has_soc',False))} "
                  f"idp.soc_complex_doubling={bool(getattr(self.idp,'soc_complex_doubling',True))} "
                  f"factor={self._soc_factor()} reduced_matrix_element={self.idp.reduced_matrix_element}")
            print("=" * 80)

        # 注意：这里保留你原来的 with_edge_vectors 调用
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)

        # ---------------- Not decompose: RME -> HR ----------------
        if not self.decompose:
            # Edge
            for k_i, opairtype in enumerate(self.idp.orbpairtype_maps.keys()):
                l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                n_rme = (2 * l1 + 1) * (2 * l2 + 1)
                sli = self.idp.orbpairtype_maps[opairtype]
                width = sli.stop - sli.start

                if do_debug and k_i < self.debug_max_pairtypes:
                    self._debug_print_pairtype("EDGE RME->HR", opairtype, width, n_rme)
                    self._debug_chunk_norms("EDGE RME(in)", data[self.edge_field], sli, n_rme)

                rme = data[self.edge_field][:, sli]
                # 关键：-1 就是 n_chunk (= n_pair * factor)
                rme = rme.reshape(n_edge, -1, n_rme).transpose(1, 2)  # (N, n_rme, n_chunk)

                HR = torch.sum(
                    self.cgbasis[opairtype][None, :, :, :, None] * rme[:, None, None, :, :],
                    dim=-2
                )  # (N, nL, nR, n_chunk)

                HR = HR.permute(0, 3, 1, 2).reshape(n_edge, -1)
                data[self.edge_field][:, sli] = HR

                if do_debug and k_i < self.debug_max_pairtypes:
                    self._debug_chunk_norms("EDGE HR(out)", data[self.edge_field], sli, n_rme)

            # Node
            if not self.overlap:
                for k_i, opairtype in enumerate(self.idp.orbpairtype_maps.keys()):
                    l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                    n_rme = (2 * l1 + 1) * (2 * l2 + 1)
                    sli = self.idp.orbpairtype_maps[opairtype]
                    width = sli.stop - sli.start

                    if do_debug and k_i < self.debug_max_pairtypes:
                        self._debug_print_pairtype("NODE RME->HR", opairtype, width, n_rme)
                        self._debug_chunk_norms("NODE RME(in)", data[self.node_field], sli, n_rme)

                    rme = data[self.node_field][:, sli]
                    rme = rme.reshape(n_node, -1, n_rme).transpose(1, 2)

                    HR = torch.sum(
                        self.cgbasis[opairtype][None, :, :, :, None] * rme[:, None, None, :, :],
                        dim=-2
                    )
                    HR = HR.permute(0, 3, 1, 2).reshape(n_node, -1)
                    data[self.node_field][:, sli] = HR

                    if do_debug and k_i < self.debug_max_pairtypes:
                        self._debug_chunk_norms("NODE HR(out)", data[self.node_field], sli, n_rme)

        # ---------------- Decompose: HR -> RME ----------------
        else:
            # Edge
            for k_i, opairtype in enumerate(self.idp.orbpairtype_maps.keys()):
                l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                nL, nR = 2 * l1 + 1, 2 * l2 + 1
                n_rme = nL * nR
                sli = self.idp.orbpairtype_maps[opairtype]
                width = sli.stop - sli.start

                if do_debug and k_i < self.debug_max_pairtypes:
                    self._debug_print_pairtype("EDGE HR->RME", opairtype, width, n_rme)
                    self._debug_chunk_norms("EDGE HR(in)", data[self.edge_field], sli, n_rme)

                HR = data[self.edge_field][:, sli]
                HR = HR.reshape(n_edge, -1, nL, nR)  # (N, n_chunk, nL, nR)

                if self.rotation:
                    angle = xyz_to_angles(data[AtomicDataDict.EDGE_VECTORS_KEY][:, [1, 2, 0]])
                    rot_mat_L = wigner_D(int(l1), angle[0], angle[1], torch.zeros_like(angle[0]))
                    rot_mat_R = wigner_D(int(l2), angle[0], angle[1], torch.zeros_like(angle[0]))
                    HR = torch.einsum("nml, nqmo, nok -> nlkq", rot_mat_L, HR, rot_mat_R)
                else:
                    HR = HR.permute(0, 2, 3, 1)  # (N, nL, nR, n_chunk)

                rme = torch.sum(
                    self.cgbasis[opairtype][None, :, :, :, None] * HR[:, :, :, None, :],
                    dim=(1, 2)
                )  # (N, n_rme, n_chunk)
                rme = rme.transpose(1, 2).reshape(n_edge, -1)
                data[self.edge_field][:, sli] = rme

                if do_debug and k_i < self.debug_max_pairtypes:
                    self._debug_chunk_norms("EDGE RME(out)", data[self.edge_field], sli, n_rme)

            # Node
            if not self.overlap:
                for k_i, opairtype in enumerate(self.idp.orbpairtype_maps.keys()):
                    l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                    nL, nR = 2 * l1 + 1, 2 * l2 + 1
                    n_rme = nL * nR
                    sli = self.idp.orbpairtype_maps[opairtype]
                    width = sli.stop - sli.start

                    if do_debug and k_i < self.debug_max_pairtypes:
                        self._debug_print_pairtype("NODE HR->RME", opairtype, width, n_rme)
                        self._debug_chunk_norms("NODE HR(in)", data[self.node_field], sli, n_rme)

                    HR = data[self.node_field][:, sli]
                    HR = HR.reshape(n_node, -1, nL, nR).permute(0, 2, 3, 1)  # (N, nL, nR, n_chunk)

                    rme = torch.sum(
                        self.cgbasis[opairtype][None, :, :, :, None] * HR[:, :, :, None, :],
                        dim=(1, 2)
                    )
                    rme = rme.transpose(1, 2).reshape(n_node, -1)
                    data[self.node_field][:, sli] = rme

                    if do_debug and k_i < self.debug_max_pairtypes:
                        self._debug_chunk_norms("NODE RME(out)", data[self.node_field], sli, n_rme)

        # legacy SOC onsite param path (与 has_soc 无关)
        data = self._apply_soc(data, n_node=n_node)

        return data

    def _initialize_CG_basis(self, pairtype: str):
        self.cgbasis.setdefault(pairtype, None)
        l1, l2 = anglrMId[pairtype[0]], anglrMId[pairtype[2]]

        cg = []
        for l_ird in range(abs(l2 - l1), l2 + l1 + 1):
            cg.append(
                wigner_3j(int(l1), int(l2), int(l_ird), dtype=self.dtype, device=self.device) * (2 * l_ird + 1) ** 0.5
            )
        cg = torch.cat(cg, dim=-1)
        self.cgbasis[pairtype] = cg
        return cg


class SKHamiltonian_old(torch.nn.Module):
    # transform SK parameters to SK hamiltonian with E3 CG basis, strain is included.
    def __init__(
        self, 
        basis: Dict[str, Union[str, list]]=None,
        idp_sk: Union[OrbitalMapper, None]=None,
        dtype: Union[str, torch.dtype] = torch.float32, 
        device: Union[str, torch.device] = torch.device("cpu"),
        edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
        node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
        onsite: bool = False,
        strain: bool = False,
        soc: bool = False,
        **kwargs,
        ) -> None:
        super(SKHamiltonian, self).__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if isinstance(device, str):
            device = torch.device(device)
        self.dtype = dtype
        self.device = device
        self.onsite = onsite
        self.soc = soc

        if basis is not None:
            self.idp_sk = OrbitalMapper(basis, method="sktb", device=device)
            if idp_sk is not None:
                assert idp_sk.basis == self.idp_sk.basis, "The basis of idp and basis should be the same."
        else:
            assert idp_sk is not None, "Either basis or idp should be provided."
            self.idp_sk = idp_sk
        # initilize a e3 indexmapping to help putting the orbital wise blocks into atom-pair wise format
        self.idp = OrbitalMapper(self.idp_sk.basis, method="e3tb", device=device)
        self.basis = self.idp.basis
        self.cgbasis = {}
        self.strain = strain
        self.edge_field = edge_field
        self.node_field = node_field

        self.idp_sk.get_orbpair_maps()
        self.idp_sk.get_skonsite_maps()
        self.idp.get_orbpair_maps()

        if self.soc:
            self.idp_sk.get_sksoc_maps()
            self.idp.get_orbpair_soc_maps()

        pairtypes = self.idp_sk.orbpairtype_maps.keys()
        
        for pairtype in pairtypes:
            # self.cgbasis.setdefault(pairtype, None)
            cg = self._initialize_CG_basis(pairtype)
            self.cgbasis[pairtype] = cg

        self.sk2irs = {
            's-s': torch.tensor([[1.]], dtype=self.dtype, device=self.device),
            's-p': torch.tensor([[1.]], dtype=self.dtype, device=self.device),
            's-d': torch.tensor([[1.]], dtype=self.dtype, device=self.device),
            'p-s': torch.tensor([[1.]], dtype=self.dtype, device=self.device),
            'p-p': torch.tensor([
                [3**0.5/3,2/3*3**0.5],[6**0.5/3,-6**0.5/3]
            ], dtype=self.dtype, device=self.device
            ),
            'p-d':torch.tensor([
                [(2/5)**0.5,(6/5)**0.5],[(3/5)**0.5,-2/5**0.5]
            ], dtype=self.dtype, device=self.device
            ),
            'd-s':torch.tensor([[1.]], dtype=self.dtype, device=self.device),
            'd-p':torch.tensor([
                [(2/5)**0.5,(6/5)**0.5],
                [(3/5)**0.5,-2/5**0.5]
            ], dtype=self.dtype, device=self.device
            ),
            'd-d':torch.tensor([
                [5**0.5/5, 2*5**0.5/5, 2*5**0.5/5],
                [2*(1/14)**0.5,2*(1/14)**0.5,-4*(1/14)**0.5],
                [3*(2/35)**0.5,-4*(2/35)**0.5,(2/35)**0.5]
                ], dtype=self.dtype, device=self.device
            )
        }
        if self.soc:
            self.soc_base_matrix = {
                's':get_soc_matrix_cubic_basis(orbital='s', device=self.device, dtype=self.dtype),
                'p':get_soc_matrix_cubic_basis(orbital='p', device=self.device, dtype=self.dtype),
                'd':get_soc_matrix_cubic_basis(orbital='d', device=self.device, dtype=self.dtype)
            }
            self.cdtype =  float2comlex(self.dtype)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # transform sk parameters to irreducible matrix element

        assert data[self.edge_field].shape[1] == self.idp_sk.reduced_matrix_element
        if self.onsite:
            assert data[self.node_field].shape[1] == self.idp_sk.n_onsite_Es
            n_node = data[self.node_field].shape[0]
            
        n_edge = data[self.edge_field].shape[0]
        

        edge_features = data[self.edge_field].clone()
        data[self.edge_field] = torch.zeros((n_edge, self.idp.reduced_matrix_element), dtype=self.dtype, device=self.device)

        # for hopping blocks
        for opairtype in self.idp_sk.orbpairtype_maps.keys():
            l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
            n_skp = min(l1, l2)+1 # number of reduced matrix element
            skparam = edge_features[:, self.idp_sk.orbpairtype_maps[opairtype]].reshape(n_edge, -1, n_skp)
            rme = skparam @ self.sk2irs[opairtype].T # shape (N, n_pair, n_rme)
            rme = rme.transpose(1,2) # shape (N, n_rme, n_pair)
            
            H_z = torch.sum(self.cgbasis[opairtype][None,:,:,:,None] * \
                rme[:,None, None, :, :], dim=-2) # shape (N, 2l1+1, 2l2+1, n_pair)
            
            # rotation
            # when get the angle, the xyz vector should be transformed to yzx.
            angle = xyz_to_angles(data[AtomicDataDict.EDGE_VECTORS_KEY][:,[1,2,0]]) # (tensor(N), tensor(N))
            # The roataion matrix is SO3 rotation, therefore Irreps(l,1), is used here.
            rot_mat_L = wigner_D(int(l1), angle[0], angle[1], torch.zeros_like(angle[0]))
            rot_mat_R = wigner_D(int(l2), angle[0], angle[1], torch.zeros_like(angle[0]))
            # rot_mat_L = Irrep(int(l1), 1).D_from_angles(angle[0].cpu(), angle[1].cpu(), torch.tensor(0., dtype=self.dtype)).to(self.device) # tensor(N, 2l1+1, 2l1+1)
            # rot_mat_R = Irrep(int(l2), 1).D_from_angles(angle[0].cpu(), angle[1].cpu(), torch.tensor(0., dtype=self.dtype)).to(self.device) # tensor(N, 2l2+1, 2l2+1)
            
            # Here The string to control einsum is important, the order of the index should be the same as the order of the tensor
            # H_z = torch.einsum("nlm, nmoq, nko -> nqlk", rot_mat_L, H_z, rot_mat_R) # shape (N, n_pair, 2l1+1, 2l2+1)
            HR = torch.einsum("nlm, nmoq, nko -> nqlk", rot_mat_L, H_z, rot_mat_R).reshape(n_edge, -1) # shape (N, n_pair * 2l2+1 * 2l2+1)
            
            if l1 < l2:
                HR = HR * (-1)**(l1+l2)
            
            data[self.edge_field][:, self.idp.orbpairtype_maps[opairtype]] = HR

        # compute onsite blocks
        if self.onsite:
            node_feature = data[self.node_field].clone()
            data[self.node_field] = torch.zeros(n_node, self.idp.reduced_matrix_element, dtype=self.dtype, device=self.device)

            for orbtype in self.idp_sk.skonsitetype_maps.keys():
                # currently, "a-b" and "b-a" orbital pair are computed seperately, it is able to combined further
                # for better performance
                l = anglrMId[re.findall(r"[a-z]", orbtype)[0]]

                skparam = node_feature[:, self.idp_sk.skonsitetype_maps[orbtype]].reshape(n_node, -1, 1)
                HR = torch.eye(2*l+1, dtype=self.dtype, device=self.device)[None, None, :, :] * skparam[:,:, None, :] # shape (N, n_pair, 2l1+1, 2l2+1)
                # the onsite block doesnot have rotation

                data[self.node_field][:, self.idp.orbpairtype_maps[orbtype+"-"+orbtype]] = HR.reshape(n_node, -1)

        if self.soc:
            assert data[AtomicDataDict.NODE_SOC_SWITCH_KEY].all(), "The SOC switch is not turned on in data by soc is set to True."
            soc_feature = data[AtomicDataDict.NODE_SOC_KEY]
            data[AtomicDataDict.NODE_SOC_KEY] = torch.zeros(n_node, self.idp.reduced_soc_matrix_elemet, dtype= self.cdtype, device=self.device)
            for otype in self.idp_sk.sksoc_maps.keys():
                lsymbol = re.findall(r"[a-z]", otype)[0]
                l = anglrMId[lsymbol]
                socparam = soc_feature[:, self.idp_sk.sksoc_maps[otype]].reshape(n_node, -1, 1)
                HR = self.soc_base_matrix[lsymbol][None, None, :, :] * socparam[:,:, None, :]
                HR_upup_updn = HR[:,:,0:2*l+1,:]
                data[AtomicDataDict.NODE_SOC_KEY][:, self.idp.orbpair_soc_maps[otype+"-"+otype]] = HR_upup_updn.reshape(n_node, -1)

        # compute if strain effect is included
        # this is a little wired operation, since it acting on somekind of a edge(strain env) feature, and summed up to return a node feature.
        if self.strain:
            n_onsitenv = len(data[AtomicDataDict.ONSITENV_FEATURES_KEY])
            for opairtype in self.idp.orbpairtype_maps.keys(): # save all env direction and pair direction like sp and ps, but only get sp
                l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                # opairtype = opair[1]+"-"+opair[4]
                n_skp = min(l1, l2)+1 # number of reduced matrix element
                skparam = data[AtomicDataDict.ONSITENV_FEATURES_KEY][:, self.idp_sk.orbpairtype_maps[opairtype]].reshape(n_onsitenv, -1, n_skp)
                rme = skparam @ self.sk2irs[opairtype].T # shape (N, n_pair, n_rme)
                rme = rme.transpose(1,2) # shape (N, n_rme, n_pair)

                H_z = torch.sum(self.cgbasis[opairtype][None,:,:,:,None] * \
                    rme[:,None, None, :, :], dim=-2) # shape (N, 2l1+1, 2l2+1, n_pair)
                
                angle = xyz_to_angles(data[AtomicDataDict.ONSITENV_VECTORS_KEY][:,[1,2,0]]) # (tensor(N), tensor(N))
                rot_mat_L = wigner_D(int(l1), angle[0], angle[1], torch.zeros_like(angle[0]))
                rot_mat_R = wigner_D(int(l2), angle[0], angle[1], torch.zeros_like(angle[0]))
                # rot_mat_L = Irrep(int(l1), 1).D_from_angles(angle[0].cpu(), angle[1].cpu(), torch.tensor(0., dtype=self.dtype)).to(self.device) # tensor(N, 2l1+1, 2l1+1)
                # rot_mat_R = Irrep(int(l2), 1).D_from_angles(angle[0].cpu(), angle[1].cpu(), torch.tensor(0., dtype=self.dtype)).to(self.device) # tensor(N, 2l2+1, 2l2+1)

                HR = torch.einsum("nlm, nmoq, nko -> nqlk", rot_mat_L, H_z, rot_mat_R) # shape (N, n_pair, 2l1+1, 2l2+1)

                HR = scatter(src=HR, index=data[AtomicDataDict.ONSITENV_INDEX_KEY][0], dim=0, reduce="sum") # shape (n_node, n_pair, 2l1+1, 2l2+1)
                # A-B o1-o2 (A-B o2-o1)= (B-A o1-o2)
                
                data[self.node_field][:, self.idp.orbpairtype_maps[opairtype]] += HR.flatten(1, len(HR.shape)-1) # the index type [node/pair] should align with the index of for loop
            
        return data

    def _initialize_CG_basis(self, pairtype: str):
        """
        The function initializes a Clebsch-Gordan basis for a given pair type.
        
        :param pairtype: The parameter "pairtype" is a string that represents a pair of angular momentum
        quantum numbers. It is expected to have a length of 3, where the first and third characters
        represent the angular momentum quantum numbers of two particles, and the second character
        represents the type of interaction between the particles
        :type pairtype: str
        :return: the CG basis, which is a tensor containing the Clebsch-Gordan coefficients for the given
        pairtype.
        """
        

        irs_index = {
            's-s': [0],
            's-p': [1],
            's-d': [2],
            's-f': [3],
            'p-s': [1],
            'p-p': [0,6],
            'p-d': [1,11],
            'd-s': [2],
            'd-p': [1,11],
            'd-d': [0,6,20]
        }

        l1, l2 = anglrMId[pairtype[0]], anglrMId[pairtype[2]]
        assert l1 <=2 and l2 <=2, "Only support l<=2, ie. s, p, d orbitals at present."
        cg = []
        for l_ird in range(abs(l2-l1), l2+l1+1):
            cg.append(wigner_3j(int(l1), int(l2), int(l_ird), dtype=self.dtype, device=self.device) * (2*l_ird+1)**0.5)
        
        cg = torch.cat(cg, dim=-1)[:,:,irs_index[pairtype]]
        
        return cg
    
class SKHamiltonian(torch.nn.Module):
    # transform SK parameters to SK hamiltonian with E3 CG basis, strain is included.
    def __init__(
        self, 
        basis: Dict[str, Union[str, list]]=None,
        idp_sk: Union[OrbitalMapper, None]=None,
        dtype: Union[str, torch.dtype] = torch.float32, 
        device: Union[str, torch.device] = torch.device("cpu"),
        edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
        node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
        onsite: bool = False,
        strain: bool = False,
        soc: bool = False,
        **kwargs,
        ) -> None:
        super(SKHamiltonian, self).__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if isinstance(device, str):
            device = torch.device(device)
        self.dtype = dtype
        self.device = device
        self.onsite = onsite
        self.soc = soc

        if basis is not None:
            self.idp_sk = OrbitalMapper(basis, method="sktb", device=device)
            if idp_sk is not None:
                assert idp_sk.basis == self.idp_sk.basis, "The basis of idp and basis should be the same."
        else:
            assert idp_sk is not None, "Either basis or idp should be provided."
            self.idp_sk = idp_sk
        # initilize a e3 indexmapping to help putting the orbital wise blocks into atom-pair wise format
        self.idp = OrbitalMapper(self.idp_sk.basis, method="e3tb", device=device)
        self.basis = self.idp.basis
        self.skbasis = {}
        self.strain = strain
        self.edge_field = edge_field
        self.node_field = node_field

        self.idp_sk.get_orbpair_maps()
        self.idp_sk.get_skonsite_maps()
        self.idp.get_orbpair_maps()

        if self.soc:
            self.idp_sk.get_sksoc_maps()
            self.idp.get_orbpair_soc_maps()

        pairtypes = self.idp_sk.orbpairtype_maps.keys()
        
        for pairtype in pairtypes:
            # self.cgbasis.setdefault(pairtype, None)
            bb = self._initialize_basis(pairtype)
            self.skbasis[pairtype] = bb

        if self.soc:
            self.soc_base_matrix = {
                's':get_soc_matrix_cubic_basis(orbital='s', device=self.device, dtype=self.dtype),
                'p':get_soc_matrix_cubic_basis(orbital='p', device=self.device, dtype=self.dtype),
                'd':get_soc_matrix_cubic_basis(orbital='d', device=self.device, dtype=self.dtype)
            }
            self.cdtype =  float2comlex(self.dtype)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # transform sk parameters to irreducible matrix element

        assert data[self.edge_field].shape[1] == self.idp_sk.reduced_matrix_element
        if self.onsite:
            assert data[self.node_field].shape[1] == self.idp_sk.n_onsite_Es
            n_node = data[self.node_field].shape[0]
            
        n_edge = data[self.edge_field].shape[0]
        

        edge_features = data[self.edge_field].clone()
        data[self.edge_field] = torch.zeros((n_edge, self.idp.reduced_matrix_element), dtype=self.dtype, device=self.device)

        # for hopping blocks
        for opairtype in self.idp_sk.orbpairtype_maps.keys():
            l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
            n_skp = min(l1, l2)+1 # number of reduced matrix element
            skparam = edge_features[:, self.idp_sk.orbpairtype_maps[opairtype]].reshape(n_edge, -1, n_skp)
            skparam = skparam.transpose(1,2) # shape (N, n_skp, n_pair)
            
            H_z = torch.sum(self.skbasis[opairtype][None,:,:,:,None] * \
                skparam[:,None, None, :, :], dim=-2) # shape (N, 2l1+1, 2l2+1, n_pair)
            
            # rotation
            # when get the angle, the xyz vector should be transformed to yzx.
            angle = xyz_to_angles(data[AtomicDataDict.EDGE_VECTORS_KEY][:,[1,2,0]]) # (tensor(N), tensor(N))
            # The roataion matrix is SO3 rotation, therefore Irreps(l,1), is used here.
            rot_mat_L = wigner_D(int(l1), angle[0], angle[1], torch.zeros_like(angle[0]))
            rot_mat_R = wigner_D(int(l2), angle[0], angle[1], torch.zeros_like(angle[0]))
            # rot_mat_L = Irrep(int(l1), 1).D_from_angles(angle[0].cpu(), angle[1].cpu(), torch.tensor(0., dtype=self.dtype)).to(self.device) # tensor(N, 2l1+1, 2l1+1)
            # rot_mat_R = Irrep(int(l2), 1).D_from_angles(angle[0].cpu(), angle[1].cpu(), torch.tensor(0., dtype=self.dtype)).to(self.device) # tensor(N, 2l2+1, 2l2+1)
            
            # Here The string to control einsum is important, the order of the index should be the same as the order of the tensor
            # H_z = torch.einsum("nlm, nmoq, nko -> nqlk", rot_mat_L, H_z, rot_mat_R) # shape (N, n_pair, 2l1+1, 2l2+1)
            HR = torch.einsum("nlm, nmoq, nko -> nqlk", rot_mat_L, H_z, rot_mat_R).reshape(n_edge, -1) # shape (N, n_pair * 2l2+1 * 2l2+1)
            
            if l1 < l2:
                HR = HR * (-1)**(l1+l2)
            
            data[self.edge_field][:, self.idp.orbpairtype_maps[opairtype]] = HR

        # compute onsite blocks
        if self.onsite:
            node_feature = data[self.node_field].clone()
            data[self.node_field] = torch.zeros(n_node, self.idp.reduced_matrix_element, dtype=self.dtype, device=self.device)

            for orbtype in self.idp_sk.skonsitetype_maps.keys():
                # currently, "a-b" and "b-a" orbital pair are computed seperately, it is able to combined further
                # for better performance
                l = anglrMId[re.findall(r"[a-z]", orbtype)[0]]

                skparam = node_feature[:, self.idp_sk.skonsitetype_maps[orbtype]].reshape(n_node, -1, 1)
                HR = torch.eye(2*l+1, dtype=self.dtype, device=self.device)[None, None, :, :] * skparam[:,:, None, :] # shape (N, n_pair, 2l1+1, 2l2+1)
                # the onsite block doesnot have rotation

                data[self.node_field][:, self.idp.orbpairtype_maps[orbtype+"-"+orbtype]] = HR.reshape(n_node, -1)

        if self.soc:
            assert data[AtomicDataDict.NODE_SOC_SWITCH_KEY].all(), "The SOC switch is not turned on in data by soc is set to True."
            soc_feature = data[AtomicDataDict.NODE_SOC_KEY]
            data[AtomicDataDict.NODE_SOC_KEY] = torch.zeros(n_node, self.idp.reduced_soc_matrix_elemet, dtype= self.cdtype, device=self.device)
            for otype in self.idp_sk.sksoc_maps.keys():
                lsymbol = re.findall(r"[a-z]", otype)[0]
                l = anglrMId[lsymbol]
                socparam = soc_feature[:, self.idp_sk.sksoc_maps[otype]].reshape(n_node, -1, 1)
                HR = self.soc_base_matrix[lsymbol][None, None, :, :] * socparam[:,:, None, :]
                HR_upup_updn = HR[:,:,0:2*l+1,:]
                data[AtomicDataDict.NODE_SOC_KEY][:, self.idp.orbpair_soc_maps[otype+"-"+otype]] = HR_upup_updn.reshape(n_node, -1)

        # compute if strain effect is included
        # this is a little wired operation, since it acting on somekind of a edge(strain env) feature, and summed up to return a node feature.
        if self.strain:
            n_onsitenv = len(data[AtomicDataDict.ONSITENV_FEATURES_KEY])
            for opairtype in self.idp.orbpairtype_maps.keys(): # save all env direction and pair direction like sp and ps, but only get sp
                l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                # opairtype = opair[1]+"-"+opair[4]
                n_skp = min(l1, l2)+1 # number of reduced matrix element
                skparam = data[AtomicDataDict.ONSITENV_FEATURES_KEY][:, self.idp_sk.orbpairtype_maps[opairtype]].reshape(n_onsitenv, -1, n_skp)
                skparam = skparam.transpose(1,2) # shape (N, n_skp, n_pair)

                H_z = torch.sum(self.skbasis[opairtype][None,:,:,:,None] * \
                    skparam[:,None, None, :, :], dim=-2) # shape (N, 2l1+1, 2l2+1, n_pair)
                
                angle = xyz_to_angles(data[AtomicDataDict.ONSITENV_VECTORS_KEY][:,[1,2,0]]) # (tensor(N), tensor(N))
                rot_mat_L = wigner_D(int(l1), angle[0], angle[1], torch.zeros_like(angle[0]))
                rot_mat_R = wigner_D(int(l2), angle[0], angle[1], torch.zeros_like(angle[0]))
                # rot_mat_L = Irrep(int(l1), 1).D_from_angles(angle[0].cpu(), angle[1].cpu(), torch.tensor(0., dtype=self.dtype)).to(self.device) # tensor(N, 2l1+1, 2l1+1)
                # rot_mat_R = Irrep(int(l2), 1).D_from_angles(angle[0].cpu(), angle[1].cpu(), torch.tensor(0., dtype=self.dtype)).to(self.device) # tensor(N, 2l2+1, 2l2+1)

                HR = torch.einsum("nlm, nmoq, nko -> nqlk", rot_mat_L, H_z, rot_mat_R) # shape (N, n_pair, 2l1+1, 2l2+1)

                HR = scatter(src=HR, index=data[AtomicDataDict.ONSITENV_INDEX_KEY][0], dim=0, reduce="sum") # shape (n_node, n_pair, 2l1+1, 2l2+1)
                # A-B o1-o2 (A-B o2-o1)= (B-A o1-o2)
                
                data[self.node_field][:, self.idp.orbpairtype_maps[opairtype]] += HR.flatten(1, len(HR.shape)-1) # the index type [node/pair] should align with the index of for loop
            
        return data

    def _initialize_basis(self, pairtype: str):
        """
        The function initializes a slater-koster used basis for a given pair type, to map the sk parameter 
        to the hamiltonian block rotated onto z-axis
        
        :param pairtype: The parameter "pairtype" is a string that represents a pair of angular momentum
        quantum numbers. It is expected to have a length of 3, where the first and third characters
        represent the angular momentum quantum numbers of two particles, and the second character
        represents the type of interaction between the particles
        :type pairtype: str
        :return: the basis, which contains a selection matrix with 0/1 as its value.
        pairtype.
        """

        basis = []
        l1, l2 = anglrMId[pairtype[0]], anglrMId[pairtype[2]]
        assert l1 <=5 and l2 <=5, "Only support l<=5, ie. s, p, d, f, g, h orbitals at present."
        for im in range(0, min(l1,l2)+1):
            mat = torch.zeros((2*l1+1, 2*l2+1), dtype=self.dtype, device=self.device)
            if im == 0:
                mat[l1,l2] = 1.
            else:
                mat[l1+im,l2+im] = 1.
                mat[l1-im, l2-im] = 1.
            basis.append(mat)
        basis = torch.stack(basis, dim=-1)
        
        return basis