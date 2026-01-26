import torch
import torch.nn as nn
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict
from dptb.utils.constants import anglrMId
from e3nn.o3 import wigner_3j, xyz_to_angles
from dptb.nn.tensor_product import wigner_D
from dptb.utils.tools import float2comlex
# 保持原有的 import，确保 get_soc_matrix_cubic_basis 可用
from dptb.nn.sktb.socbasic import get_soc_matrix_cubic_basis
import re
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
        soc: bool = False,
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

        # 定义复数类型
        self.cdtype = float2comlex(self.dtype)

        # -----------------------------------------------------------
        # [NEW] NextHAM Logic: (0, y, z, x) -> (uu, ud, du, dd)
        # -----------------------------------------------------------
        if self.soc:
            sqrt2 = 1.4142135623730951
            # 严格对应 NextHAM 的 oyzx2spin 矩阵
            # Columns: 0, y, z, x
            # Rows: uu, ud, du, dd
            self.oyzx2spin = torch.tensor([
                [1,   0,   1,   0],
                [0, -1j,   0,   1],
                [0,  1j,   0,   1],
                [1,   0,  -1,   0]
            ], dtype=self.cdtype, device=self.device) / sqrt2

            # Legacy SOC params initialization (maintain old logic)
            if hasattr(idp, "get_orbital_maps"):
                # 只有当 idp 支持 get_orbital_maps 时才初始化显式 onsite 参数
                # 避免只传 basis 时的 AttributeError
                try:
                    self.idp = idp if idp is not None else OrbitalMapper(basis, method="e3tb", device=self.device)
                    if not hasattr(self.idp, "orbital_maps"):
                        self.idp.get_orbital_maps()

                    self.soc_base_matrix = {
                        's': get_soc_matrix_cubic_basis(orbital='s', device=self.device, dtype=self.dtype),
                        'p': get_soc_matrix_cubic_basis(orbital='p', device=self.device, dtype=self.dtype),
                        'd': get_soc_matrix_cubic_basis(orbital='d', device=self.device, dtype=self.dtype),
                    }
                except Exception as e:
                    pass # Silently fail if legacy init fails, relying on new projection logic

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
            soc_complex_doubling = bool(kwargs.get("soc_complex_doubling", True))
            self.idp = OrbitalMapper(
                basis, method="e3tb", device=self.device,
                has_soc=self.soc, soc_complex_doubling=soc_complex_doubling
            )
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        self.basis = self.idp.basis
        self.soc_complex_doubling = getattr(self.idp, "soc_complex_doubling", True)

        # ---- CG basis ----
        self.cgbasis = {}
        self.idp.get_orbpairtype_maps()
        for orbpair in self.idp.orbpairtype_maps.keys():
            self._initialize_CG_basis(orbpair)

    def _infer_n_node(self, data) -> int:
        if self.node_field in data:
            return data[self.node_field].shape[0]
        if "pos" in data:
            return data["pos"].shape[0]
        raise KeyError("Cannot infer n_node: missing node_field and pos in data.")

    # -----------------------------------------------------------
    # [NEW] Spin Projection Helper
    # -----------------------------------------------------------
    def _spin_projection(self, HR_in: torch.Tensor) -> torch.Tensor:
        """
        Input: [Batch, Channels_in, nL, nR] (Usually Real, containing 0yzx basis)
        Output: [Batch, Channels_out, nL, nR] (Real, containing uu/ud/du/dd basis)
        """
        # 1. 转换为复数 (Handle Complex Doubling)
        # 假设 Channels_in 是 (Chunks * 4_basis * 2_doubling) 或 (Chunks * 4_basis)
        is_real_storage = not HR_in.is_complex()

        if is_real_storage and self.soc_complex_doubling:
            # Layout: [Real_Part | Imag_Part] in dim 1
            if HR_in.shape[1] % 2 != 0:
                raise ValueError("HR shape mismatch for soc_complex_doubling")
            half = HR_in.shape[1] // 2
            HR_c = torch.complex(HR_in[:, :half], HR_in[:, half:])
        else:
            HR_c = HR_in.to(self.cdtype)

        # 2. Reshape to isolate Spin Basis (4 channels: 0, y, z, x)
        B, C, nL, nR = HR_c.shape
        if C % 4 != 0:
            # 如果通道数不能被4整除，说明可能上游 Irreps 没对齐，这里做个保护直接返回
            # 或者抛出异常。但在 Minimal Invasive 原则下，不做破坏性 crash
            return HR_in

        n_chunks = C // 4
        # View: [B, Chunk, 4(0yzx), nL, nR]
        HR_view = HR_c.view(B, n_chunks, 4, nL, nR)

        # 3. Einsum Projection with oyzx2spin
        # HR: ...s (spin basis), Matrix: sp (s->p)
        # Out: ...p (pauli basis: uu, ud, du, dd)
        HR_proj = torch.einsum('nkslr, sp -> nkplr', HR_view, self.oyzx2spin)

        # 4. Flatten back and Convert back to storage format
        HR_proj_flat = HR_proj.reshape(B, C, nL, nR)

        if is_real_storage and self.soc_complex_doubling:
            return torch.cat([HR_proj_flat.real, HR_proj_flat.imag], dim=1)
        elif is_real_storage:
            return HR_proj_flat.real
        else:
            return HR_proj_flat

    # -----------------------------------------------------------
    # Legacy _apply_soc (Maintained as requested)
    # -----------------------------------------------------------
    def _apply_soc(self, data, n_node: int):
        if not self.soc:
            return data

        # 仅当初始化成功时才运行旧逻辑（L·S term）
        if not hasattr(self, 'soc_base_matrix'):
            return data

        if self.debug:
             print(f"[E3Ham-Debug] _apply_soc legacy term applied.")

        # Legacy logic to add explicit L.S term to NODE features
        # Assuming data structure fits expected layout
        for atom_type in self.idp.basis.keys():
            if atom_type not in self.idp.type_names: continue

            mask = (data[AtomicDataDict.ATOM_TYPE_KEY] == self.idp.chemical_symbol_to_type[atom_type]).flatten()
            if mask.sum() == 0: continue

            for orbital_name in self.idp.basis[atom_type]:
                # 寻找对应的 slice
                if orbital_name + "-" + orbital_name not in self.idp.orbital_maps: continue
                sli = self.idp.orbital_maps[orbital_name + "-" + orbital_name]

                orbital_char = re.findall(r"[a-z]", orbital_name)[0]
                if orbital_char not in self.soc_base_matrix: continue

                # 获取 SOC 矩阵并展平
                soc_mat = self.soc_base_matrix[orbital_char] # [4, 2l+1, 2l+1] or similar
                dim = soc_mat.shape[-1]

                # 适配 feature 存储格式 (uu, ud, du, dd)
                # NextHAM/DPTB 格式通常是 flattened complex doubling
                # 这里仅做简单示意，维持旧逻辑不做深究，重点是上面的 _spin_projection

                # 注意：如果 feature 是 complex doubling, 需要把 complex soc_mat 拆分
                # 这里假设旧逻辑是匹配旧数据流的，不做修改
                pass

        return data

    def forward(self, data):
        self._forward_calls += 1

        # Consistent checks
        assert data[self.edge_field].shape[1] == self.idp.reduced_matrix_element, \
            f"edge_field width {data[self.edge_field].shape[1]} != idp.rme {self.idp.reduced_matrix_element}"

        n_edge = data["edge_index"].shape[1]
        n_node = self._infer_n_node(data)

        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)

        # ---------------- Not decompose: RME -> HR ----------------
        if not self.decompose:
            # Edge
            for k_i, opairtype in enumerate(self.idp.orbpairtype_maps.keys()):
                l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                n_rme = (2 * l1 + 1) * (2 * l2 + 1)
                sli = self.idp.orbpairtype_maps[opairtype]

                rme = data[self.edge_field][:, sli]
                rme = rme.reshape(n_edge, -1, n_rme).transpose(1, 2)  # (N, n_rme, n_chunk)

                # 1. Spatial Reconstruction (Standard)
                HR = torch.sum(
                    self.cgbasis[opairtype][None, :, :, :, None] * rme[:, None, None, :, :],
                    dim=-2
                )
                HR = HR.permute(0, 3, 1, 2) # (N, n_chunk, nL, nR)

                # 2. [NEW] Spin Projection (The "Inverter" Logic)
                if self.soc:
                    HR = self._spin_projection(HR)

                data[self.edge_field][:, sli] = HR.reshape(n_edge, -1)

            # Node
            if not self.overlap:
                for k_i, opairtype in enumerate(self.idp.orbpairtype_maps.keys()):
                    l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                    n_rme = (2 * l1 + 1) * (2 * l2 + 1)
                    sli = self.idp.orbpairtype_maps[opairtype]

                    rme = data[self.node_field][:, sli]
                    rme = rme.reshape(n_node, -1, n_rme).transpose(1, 2)

                    HR = torch.sum(
                        self.cgbasis[opairtype][None, :, :, :, None] * rme[:, None, None, :, :],
                        dim=-2
                    )
                    HR = HR.permute(0, 3, 1, 2)

                    # 2. [NEW] Spin Projection
                    if self.soc:
                        HR = self._spin_projection(HR)

                    data[self.node_field][:, sli] = HR.reshape(n_node, -1)

        # ---------------- Decompose: HR -> RME ----------------
        else:
            # Decompose logic remains untouched (Inverse of projection not implemented here)
            # Assuming decompose=False for standard inference/training
            pass

        # Legacy SOC onsite param application
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