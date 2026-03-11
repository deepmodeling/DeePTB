import os
import re
import h5py
import numpy as np
from ase.io import read
import ase
from typing import Union, Optional
import torch
import logging
from copy import deepcopy

log = logging.getLogger(__name__)

from dptb.data import AtomicData, AtomicDataDict, block_to_feature
from dptb.nn.energy import Eigenvalues
from dptb.utils.argcheck import get_cutoffs_from_model_options
from dptb.utils.constants import Boltzmann, eV2J, anglrMId


class ElecStruCal(object):
    """
    ElecStruCal:
      data(ase.Atoms/AtomicData/str) -> (optional inject H/S) -> idp -> model -> HR2HK -> eig

    支持的推理 tag（kpath_kwargs 透传到 get_eigs）：
      - override_overlap: 从文件覆盖 Overlap
      - add_h0: 从文件注入 H0，并在推理时做 H = H0 + ΔH（SOC+NextHAM 会对 ΔH 做 uu.real -> (uu.real, dd.real) 的过滤）
      - override_full_h: 直接从文件载入 full H（跳过模型，优先级最高）

    新增用于误差归因的两个 tag（需要与 add_h0 配合，且在 add_h0 之后生效）：
      - override_full_h_uureal:
            在 add_h0 得到 H 之后，再载入 Full-H：
              * 仅使用 Full-H 的 uu.real 来构造 Δ = FullH_uu.real - H0_uu.real
              * uu.real := H0_uu.real + Δ （等价于 uu.real = FullH_uu.real）
              * dd.real := H0_dd.real + Δ （由 uu.real 的 Δ 等价对称推导，不直接使用 FullH_dd.real）
            其余分量保持当前（通常来自 H0 或其它设置），以避免引入额外自由度污染误差归因。
      - override_full_h_wo_uureal:
            在 add_h0 得到 H 之后，再载入 Full-H，挑选 Full-H 中 “除了 uu.real + dd.real 之外的所有分量” 进行替换
            （即保留模型给出的 uu/dd 实部，其它全部用 Full-H）
    """

    def __init__(
            self,
            model: torch.nn.Module,
            device: Union[str, torch.device] = None
    ):
        if device is None:
            device = model.device
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model = model
        self.model.eval()
        self.overlap = hasattr(model, 'overlap')

        if not self.model.transform:
            log.error('The model.transform is not True, please check the model.')
            raise RuntimeError('The model.transform is not True, please check the model.')

        # eigv init
        if self.overlap:
            self.eigv = Eigenvalues(
                idp=model.idp,
                device=self.device,
                s_edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                s_node_field=AtomicDataDict.NODE_OVERLAP_KEY,
                s_out_field=AtomicDataDict.OVERLAP_KEY,
                dtype=model.dtype,
            )
        else:
            self.eigv = Eigenvalues(
                idp=model.idp,
                device=self.device,
                dtype=model.dtype,
            )

        r_max, er_max, oer_max = get_cutoffs_from_model_options(model.model_options)
        self.cutoffs = {'r_max': r_max, 'er_max': er_max, 'oer_max': oer_max}

    # =========================================================
    # helpers: H5 read
    # =========================================================
    @staticmethod
    def _open_h5_first_block(path: str):
        """
        打开 h5 并返回 (h5file, block)。
        block 默认取 key "0" 否则取 "1"。
        调用者负责 close file。
        """
        f = h5py.File(path, "r")
        if "0" in f:
            blk = f["0"]
        else:
            blk = f["1"]
        return f, blk

    # =========================================================
    # helpers: build mask_uureal if missing
    # =========================================================
    def _ensure_mask_uureal(self, idp):
        """
        如果 checkpoint 的 idp 没保存 mask_uureal，这里按 orbpair_maps slice layout 动态补齐。

        规则：每个 orbpair slice 的前 base_dim = (2l_i+1)(2l_j+1) 对应 uu_real（SOC NextHAM 约定）。
        """
        if hasattr(idp, "mask_uureal") and idp.mask_uureal is not None:
            return

        if not hasattr(idp, "orbpair_maps") or idp.orbpair_maps is None:
            if hasattr(idp, "get_orbpair_maps"):
                idp.get_orbpair_maps()
            else:
                raise RuntimeError("idp has no orbpair_maps and cannot build it.")

        rme = getattr(idp, "reduced_matrix_element", None)
        if rme is None:
            raise RuntimeError("idp has no reduced_matrix_element; cannot build mask_uureal.")

        mask = torch.zeros(rme, dtype=torch.bool, device=torch.device("cpu"))

        items = list(idp.orbpair_maps.items())
        items.sort(key=lambda kv: kv[1].start)

        for k, sli in items:
            io, jo = k.split("-")
            il = anglrMId[re.findall(r"[a-z]", io)[0]]
            jl = anglrMId[re.findall(r"[a-z]", jo)[0]]
            base_dim = (2 * il + 1) * (2 * jl + 1)
            mask[sli.start: sli.start + base_dim] = True

        idp.mask_uureal = mask
        log.warning(
            f"[ElecStruCal] mask_uureal not found in checkpoint; built on-the-fly: "
            f"{int(mask.sum().item())}/{mask.numel()} dims kept."
        )

    # =========================================================
    # helpers: NextHAM indices + filter delta
    # =========================================================
    def _get_nextham_uureal_ddreal_indices(self, idp, device: torch.device):
        """
        返回 (uu_idx, dd_idx)，并缓存到 idp：
          uu_idx: uu.real 的 indices（升序）
          dd_idx: dd.real 的 indices（升序），与 uu_idx 一一对应（同 base_dim 分块）

        假设 slice 内 “实部 half” 的布局为：
          [uu, ud, du, dd] each base_dim
        因此 dd.real 对应：
          sli.start + 3*base_dim : sli.start + 4*base_dim
        """
        self._ensure_mask_uureal(idp)

        if hasattr(idp, "_nextham_uureal_idx") and hasattr(idp, "_nextham_ddreal_idx"):
            uu_idx = idp._nextham_uureal_idx.to(device)
            dd_idx = idp._nextham_ddreal_idx.to(device)
            return uu_idx, dd_idx

        uu_idx = torch.nonzero(idp.mask_uureal, as_tuple=False).flatten().to(device)

        items = list(idp.orbpair_maps.items())
        items.sort(key=lambda kv: kv[1].start)

        dd_idx_list = []
        for k, sli in items:
            io, jo = k.split("-")
            il = anglrMId[re.findall(r"[a-z]", io)[0]]
            jl = anglrMId[re.findall(r"[a-z]", jo)[0]]
            base_dim = (2 * il + 1) * (2 * jl + 1)

            dd_start = int(sli.start + 3 * base_dim)
            dd_stop = int(sli.start + 4 * base_dim)
            dd_idx_list.append(torch.arange(dd_start, dd_stop, dtype=torch.long, device=device))

        dd_idx = torch.cat(dd_idx_list, dim=0)

        if uu_idx.numel() != dd_idx.numel():
            raise RuntimeError(
                f"[ElecStruCal] uu_real_idx.size != dd_real_idx.size: {uu_idx.numel()} vs {dd_idx.numel()}.\n"
                f"Likely orbpair_maps layout mismatched with mask_uureal assumption."
            )

        idp._nextham_uureal_idx = uu_idx.detach().cpu()
        idp._nextham_ddreal_idx = dd_idx.detach().cpu()
        return uu_idx, dd_idx

    def _nextham_filter_delta_soc(self, delta_feat: torch.Tensor, idp) -> torch.Tensor:
        """
        NextHAM 推理 delta 规则（SOC）：
          - 只取 uu.real 的 delta
          - 同时写到 uu.real 和 dd.real
          - 其它（ud/du、imag、dd 的其他部分）全部置零
        """
        uu_idx, dd_idx = self._get_nextham_uureal_ddreal_indices(idp, device=delta_feat.device)
        out = torch.zeros_like(delta_feat)

        if delta_feat.is_complex():
            uu_val = delta_feat[..., uu_idx].real
            uu_val_c = torch.complex(uu_val, torch.zeros_like(uu_val))
            out[..., uu_idx] = uu_val_c
            out[..., dd_idx] = uu_val_c
        else:
            uu_val = delta_feat[..., uu_idx]
            out[..., uu_idx] = uu_val
            out[..., dd_idx] = uu_val
        return out

    # =========================================================
    # helpers: apply override_full_h_uureal / override_full_h_wo_uureal
    # =========================================================
    def _apply_fullh_uureal_sym_after_add_h0(
            self,
            feat: torch.Tensor,       # 当前已经 add_h0 后的 H
            full_feat: torch.Tensor,  # Full-H 对应特征
            h0_feat: torch.Tensor,    # H0 对应特征（add_h0 注入的那份）
            uu_idx: torch.Tensor,
            dd_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        override_full_h_uureal 的正确实现（符合 NextHAM/add_h0 范式）：
          - 只用 Full-H 的 uu.real 计算 Δ = FullH_uu.real - H0_uu.real
          - uu.real = H0_uu.real + Δ （即 uu.real = FullH_uu.real）
          - dd.real = H0_dd.real + Δ （由同一份 Δ 等价对称推导，不直接使用 FullH_dd.real）
          - 其它分量不动（ud/du、imag 等保持当前 feat）
        """
        if feat.shape != full_feat.shape or feat.shape != h0_feat.shape:
            raise RuntimeError(
                f"[ElecStruCal] Feature shape mismatch: feat={feat.shape}, full={full_feat.shape}, h0={h0_feat.shape}"
            )

        # dtype/device align
        if full_feat.device != feat.device:
            full_feat = full_feat.to(feat.device)
        if h0_feat.device != feat.device:
            h0_feat = h0_feat.to(feat.device)
        if full_feat.dtype != feat.dtype:
            full_feat = full_feat.to(feat.dtype)
        if h0_feat.dtype != feat.dtype:
            h0_feat = h0_feat.to(feat.dtype)

        if feat.is_complex():
            delta = full_feat[..., uu_idx].real - h0_feat[..., uu_idx].real
            new_uu_real = h0_feat[..., uu_idx].real + delta   # == full uu.real
            new_dd_real = h0_feat[..., dd_idx].real + delta   # symmetry from uu.real delta
            feat[..., uu_idx] = torch.complex(new_uu_real, feat[..., uu_idx].imag)
            feat[..., dd_idx] = torch.complex(new_dd_real, feat[..., dd_idx].imag)
        else:
            delta = full_feat[..., uu_idx] - h0_feat[..., uu_idx]
            feat[..., uu_idx] = h0_feat[..., uu_idx] + delta  # == full uu.real
            feat[..., dd_idx] = h0_feat[..., dd_idx] + delta  # dd.real = H0_dd.real + Δ

        return feat

    def _apply_fullh_patch_after_add_h0(
            self,
            feat: torch.Tensor,
            full_feat: torch.Tensor,
            uu_idx: torch.Tensor,
            dd_idx: torch.Tensor,
            mode: str,
    ) -> torch.Tensor:
        """
        mode:
          - "wo_uureal": 除 uu.real + dd.real 外全部用 Full-H 覆盖（即保留 uu/dd real）

        注意：uureal 模式不要用这个函数（会直接替换 dd.real，引入额外自由度）。
              override_full_h_uureal 应使用 _apply_fullh_uureal_sym_after_add_h0().
        """
        if feat.shape != full_feat.shape:
            raise RuntimeError(f"[ElecStruCal] Feature shape mismatch: {feat.shape} vs {full_feat.shape}")

        # dtype/device align
        if full_feat.device != feat.device:
            full_feat = full_feat.to(feat.device)
        if full_feat.dtype != feat.dtype:
            full_feat = full_feat.to(feat.dtype)

        dim = feat.shape[-1]
        keep_mask = torch.ones(dim, dtype=torch.bool, device=feat.device)
        keep_mask[uu_idx] = False
        keep_mask[dd_idx] = False  # keep_mask == other parts

        if mode == "wo_uureal":
            if feat.is_complex():
                other_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
                feat[..., other_idx] = full_feat[..., other_idx]
                # uu/dd：保留 real（来自当前 feat），imag 用 Full-H
                feat[..., uu_idx] = torch.complex(feat[..., uu_idx].real, full_feat[..., uu_idx].imag)
                feat[..., dd_idx] = torch.complex(feat[..., dd_idx].real, full_feat[..., dd_idx].imag)
            else:
                feat[..., keep_mask] = full_feat[..., keep_mask]
            return feat

        raise ValueError(f"Unknown mode={mode}, expected 'wo_uureal'.")

    # =========================================================
    # get_data
    # =========================================================
    def get_data(self,
                 data: Union[AtomicData, ase.Atoms, str],
                 pbc: Union[bool, list] = None,
                 device: Union[str, torch.device] = None,
                 AtomicData_options: dict = None,
                 override_overlap: Optional[str] = None,
                 add_h0: Optional[str] = None,
                 override_full_h: Optional[str] = None,
                 override_full_h_uureal: Optional[str] = None,      # 签名兼容（逻辑在 get_eigs）
                 override_full_h_wo_uureal: Optional[str] = None    # 签名兼容（逻辑在 get_eigs）
                 ):
        """
        构造 AtomicData，并根据 tag 写入：
          - override_overlap: 覆盖 S blocks
          - add_h0: 写入 H0 blocks 到 EDGE/NODE_FEATURES_KEY（供 get_eigs 里 add back）
          - override_full_h: 写入 full H blocks 到 EDGE/NODE_FEATURES_KEY（用于完全跳过模型）
        """
        atomic_options = deepcopy(self.cutoffs)
        if pbc is not None:
            atomic_options.update({'pbc': pbc})

        if AtomicData_options is not None:
            for k in ["r_max", "er_max", "oer_max"]:
                if AtomicData_options.get(k, None) is not None and atomic_options.get(k, None) != AtomicData_options.get(k):
                    atomic_options[k] = AtomicData_options.get(k)
                    log.warning(f'Overwrite {k} with AtomicData_options[{k}]={atomic_options[k]} (dangerous).')
        else:
            if atomic_options['r_max'] is None:
                raise RuntimeError('r_max is not provided in model_options; please provide it in AtomicData_options.')

        # build AtomicData
        if isinstance(data, str):
            structase = read(data)
            data = AtomicData.from_ase(structase, **atomic_options)
        elif isinstance(data, ase.Atoms):
            data = AtomicData.from_ase(data, **atomic_options)
        elif isinstance(data, AtomicData):
            data = deepcopy(data)
            log.info('The data is already an instance of AtomicData. Use a deepcopy to avoid in-place pollution.')
        else:
            raise ValueError('data should be either a string, ase.Atoms, or AtomicData')

        # tag priority: override_full_h > add_h0
        if isinstance(override_full_h, str) and isinstance(add_h0, str):
            log.warning("[ElecStruCal] Both override_full_h and add_h0 are provided; "
                        "override_full_h will take precedence and add_h0 will be ignored.")

        overlaps_blk = None
        fullh_blk = None
        h0_blk = None

        fS = fH = None
        try:
            if isinstance(override_overlap, str):
                if not os.path.exists(override_overlap):
                    raise FileNotFoundError(f"Overlap file not found: {override_overlap}")
                fS, overlaps_blk = self._open_h5_first_block(override_overlap)

            if isinstance(override_full_h, str):
                if not os.path.exists(override_full_h):
                    raise FileNotFoundError(f"Full H file not found: {override_full_h}")
                fH, fullh_blk = self._open_h5_first_block(override_full_h)
            elif isinstance(add_h0, str):
                if not os.path.exists(add_h0):
                    raise FileNotFoundError(f"H0 file not found: {add_h0}")
                fH, h0_blk = self._open_h5_first_block(add_h0)

            if overlaps_blk is not None:
                if fullh_blk is not None:
                    block_to_feature(data, self.model.idp, blocks=fullh_blk, overlap_blocks=overlaps_blk)
                elif h0_blk is not None:
                    block_to_feature(data, self.model.idp, blocks=h0_blk, overlap_blocks=overlaps_blk)
                else:
                    block_to_feature(data, self.model.idp, blocks=False, overlap_blocks=overlaps_blk)

                # if model doesn't have overlap head, we must use generalized eig
                if not self.overlap:
                    self.eigv = Eigenvalues(
                        idp=self.model.idp,
                        device=self.device,
                        s_edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                        s_node_field=AtomicDataDict.NODE_OVERLAP_KEY,
                        s_out_field=AtomicDataDict.OVERLAP_KEY,
                        dtype=self.model.dtype,
                    )
            else:
                if fullh_blk is not None:
                    block_to_feature(data, self.model.idp, blocks=fullh_blk, overlap_blocks=False)
                elif h0_blk is not None:
                    block_to_feature(data, self.model.idp, blocks=h0_blk, overlap_blocks=False)
                else:
                    pass

        finally:
            if fS is not None:
                fS.close()
            if fH is not None:
                fH.close()

        if device is None:
            device = self.device
        data = AtomicData.to_AtomicDataDict(data.to(device))
        data = self.model.idp(data)
        return data

    # =========================================================
    # get_eigs
    # =========================================================
    def get_eigs(self,
                 data: Union[AtomicData, ase.Atoms, str],
                 klist: np.ndarray,
                 pbc: Union[bool, list] = None,
                 AtomicData_options: dict = None,
                 override_overlap: Optional[str] = None,
                 add_h0: Optional[str] = None,
                 override_full_h: Optional[str] = None,
                 override_full_h_uureal: Optional[str] = None,
                 override_full_h_wo_uureal: Optional[str] = None,
                 eig_solver: Optional[str] = None):
        """
        计算指定 klist 下 eigenvalues。

        分支逻辑：
          - override_full_h: 直接用文件里的 full H，跳过模型（最高优先级）
          - 否则：
              - 若 add_h0: 缓存 H0
              - model forward 得到 ΔH
              - SOC+add_h0 时按 NextHAM 规则过滤 ΔH（只保留 uu.real，并对称到 dd.real）
              - H = H0 + ΔH
              - 若 override_full_h_uureal / override_full_h_wo_uureal: 再读 Full-H 做分量替换
        """
        # mutual exclusion checks
        if override_full_h_uureal and override_full_h_wo_uureal:
            raise ValueError("Cannot set both override_full_h_uureal and override_full_h_wo_uureal.")

        if override_full_h and (override_full_h_uureal or override_full_h_wo_uureal):
            log.warning("[ElecStruCal] override_full_h has highest priority; "
                        "override_full_h_uureal/wo_uureal will be ignored.")

        data_in = data

        data = self.get_data(
            data=data_in, pbc=pbc, device=self.device, AtomicData_options=AtomicData_options,
            override_overlap=override_overlap, add_h0=add_h0, override_full_h=override_full_h,
            override_full_h_uureal=override_full_h_uureal,
            override_full_h_wo_uureal=override_full_h_wo_uureal,
        )

        # set kpoints
        data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor(
            [torch.as_tensor(klist, dtype=self.model.dtype, device=self.device)]
        )

        # cache overlap if override_overlap (only needed if we call model forward)
        if isinstance(override_overlap, str):
            override_overlap_edge = data[AtomicDataDict.EDGE_OVERLAP_KEY]
            override_overlap_node = data[AtomicDataDict.NODE_OVERLAP_KEY]

        # ============================
        # override_full_h branch
        # ============================
        if isinstance(override_full_h, str):
            if isinstance(add_h0, str):
                log.warning("[ElecStruCal] override_full_h is enabled, add_h0 will be ignored.")

            if data.get(AtomicDataDict.EDGE_FEATURES_KEY) is None:
                raise RuntimeError("override_full_h is set but EDGE_FEATURES_KEY not found in data.")
            if data.get(AtomicDataDict.NODE_FEATURES_KEY) is None:
                raise RuntimeError("override_full_h is set but NODE_FEATURES_KEY not found in data.")

            if self.overlap or isinstance(override_overlap, str):
                assert data.get(AtomicDataDict.EDGE_OVERLAP_KEY) is not None

            data = self.eigv(data)
            return data, data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0].detach().cpu().numpy()

        # ============================
        # normal model inference branch
        # ============================
        if isinstance(add_h0, str):
            h0_edge = data[AtomicDataDict.EDGE_FEATURES_KEY]
            h0_node = data[AtomicDataDict.NODE_FEATURES_KEY]

        # model forward -> predicted ΔH
        data = self.model(data)

        # restore override overlap (avoid being overwritten by model)
        if isinstance(override_overlap, str):
            data[AtomicDataDict.EDGE_OVERLAP_KEY] = override_overlap_edge
            data[AtomicDataDict.NODE_OVERLAP_KEY] = override_overlap_node

        # add_h0 path: filter delta in SOC NextHAM style, then add back H0
        if isinstance(add_h0, str):
            if getattr(self.model.idp, "has_soc", False):
                data[AtomicDataDict.EDGE_FEATURES_KEY] = self._nextham_filter_delta_soc(
                    data[AtomicDataDict.EDGE_FEATURES_KEY], self.model.idp
                )
                data[AtomicDataDict.NODE_FEATURES_KEY] = self._nextham_filter_delta_soc(
                    data[AtomicDataDict.NODE_FEATURES_KEY], self.model.idp
                )

            data[AtomicDataDict.EDGE_FEATURES_KEY] = h0_edge + data[AtomicDataDict.EDGE_FEATURES_KEY]
            data[AtomicDataDict.NODE_FEATURES_KEY] = h0_node + data[AtomicDataDict.NODE_FEATURES_KEY]

        # ============================
        # patch by Full-H after add_h0
        # ============================
        patch_fullh_path = override_full_h_uureal or override_full_h_wo_uureal
        if patch_fullh_path:
            if not isinstance(add_h0, str):
                raise ValueError("override_full_h_uureal / override_full_h_wo_uureal must be used with add_h0.")
            if not getattr(self.model.idp, "has_soc", False):
                raise ValueError("override_full_h_uureal / override_full_h_wo_uureal are intended for SOC mode only.")
            if not os.path.exists(patch_fullh_path):
                raise FileNotFoundError(f"Full-H file not found: {patch_fullh_path}")

            # load Full-H into another data dict (same structure, same cutoffs) for component replacement
            # NOTE: do NOT pass add_h0 here; we want pure Full-H
            full_data = self.get_data(
                data=data_in, pbc=pbc, device=self.device, AtomicData_options=AtomicData_options,
                override_overlap=override_overlap,
                add_h0=None,
                override_full_h=patch_fullh_path,
            )
            full_data[AtomicDataDict.KPOINT_KEY] = data[AtomicDataDict.KPOINT_KEY]

            # sanity check edge ordering
            if AtomicDataDict.EDGE_INDEX_KEY in data and AtomicDataDict.EDGE_INDEX_KEY in full_data:
                if data[AtomicDataDict.EDGE_INDEX_KEY].shape != full_data[AtomicDataDict.EDGE_INDEX_KEY].shape or \
                        not torch.equal(data[AtomicDataDict.EDGE_INDEX_KEY], full_data[AtomicDataDict.EDGE_INDEX_KEY]):
                    raise RuntimeError(
                        "[ElecStruCal] Edge order mismatch between main data and full-H data. "
                        "Cannot safely apply component override."
                    )

            uu_idx, dd_idx = self._get_nextham_uureal_ddreal_indices(self.model.idp, device=self.device)

            if override_full_h_uureal:
                # 正确的 uureal 归因：只用 FullH 的 uu.real 来构造 Δ，并对称推导 dd.real（不直接用 FullH_dd.real）
                data[AtomicDataDict.EDGE_FEATURES_KEY] = self._apply_fullh_uureal_sym_after_add_h0(
                    feat=data[AtomicDataDict.EDGE_FEATURES_KEY],
                    full_feat=full_data[AtomicDataDict.EDGE_FEATURES_KEY],
                    h0_feat=h0_edge,
                    uu_idx=uu_idx,
                    dd_idx=dd_idx,
                )
                data[AtomicDataDict.NODE_FEATURES_KEY] = self._apply_fullh_uureal_sym_after_add_h0(
                    feat=data[AtomicDataDict.NODE_FEATURES_KEY],
                    full_feat=full_data[AtomicDataDict.NODE_FEATURES_KEY],
                    h0_feat=h0_node,
                    uu_idx=uu_idx,
                    dd_idx=dd_idx,
                )
                log.warning(f"[ElecStruCal] Applied post-add_h0 Full-H patch mode=uureal(sym) from: {patch_fullh_path}")
            else:
                # wo_uureal：保留 uu/dd real，其它全部用 Full-H
                data[AtomicDataDict.EDGE_FEATURES_KEY] = self._apply_fullh_patch_after_add_h0(
                    feat=data[AtomicDataDict.EDGE_FEATURES_KEY],
                    full_feat=full_data[AtomicDataDict.EDGE_FEATURES_KEY],
                    uu_idx=uu_idx,
                    dd_idx=dd_idx,
                    mode="wo_uureal",
                )
                data[AtomicDataDict.NODE_FEATURES_KEY] = self._apply_fullh_patch_after_add_h0(
                    feat=data[AtomicDataDict.NODE_FEATURES_KEY],
                    full_feat=full_data[AtomicDataDict.NODE_FEATURES_KEY],
                    uu_idx=uu_idx,
                    dd_idx=dd_idx,
                    mode="wo_uureal",
                )
                log.warning(f"[ElecStruCal] Applied post-add_h0 Full-H patch mode=wo_uureal from: {patch_fullh_path}")

        if self.overlap or isinstance(override_overlap, str):
            assert data.get(AtomicDataDict.EDGE_OVERLAP_KEY) is not None

        data = self.eigv(data)
        return data, data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0].detach().cpu().numpy()

    # =========================================================
    # fermi level (unchanged)
    # =========================================================
    def get_fermi_level(self, data: Union[AtomicData, ase.Atoms, str], nel_atom: dict,
                        meshgrid: list = None, klist: np.ndarray = None, pbc: Union[bool, list] = None,
                        AtomicData_options: dict = None, q_tol: float = 1e-10, smearing_method: str = 'FD',
                        temp: float = 300, eig_solver: Optional[str] = 'torch'):

        assert meshgrid is not None or klist is not None, 'kmesh or klist should be provided.'
        assert isinstance(nel_atom, dict)

        if klist is None:
            from dptb.utils.make_kpoints import kmesh_sampling_negf
            klist, wk = kmesh_sampling_negf(meshgrid=meshgrid, is_gamma_center=True, is_time_reversal=True)
            log.info(f'KPOINTS  kmesh sampling: {klist.shape[0]} kpoints')
        else:
            wk = np.ones(klist.shape[0]) / klist.shape[0]
            log.info(f'KPOINTS  klist: {klist.shape[0]} kpoints')

        if not AtomicDataDict.ENERGY_EIGENVALUE_KEY in data:
            data, eigs = self.get_eigs(data=data, klist=klist, pbc=pbc,
                                       AtomicData_options=AtomicData_options,
                                       eig_solver=eig_solver)
            log.info('Getting eigenvalues from the model.')
        else:
            eigs = data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0].detach().cpu().numpy()

        atomtype_list = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().tolist()
        atomtype_symbols = np.asarray(self.model.idp.type_names)[atomtype_list].tolist()
        total_nel = np.array([nel_atom[s] for s in atomtype_symbols]).sum()

        if getattr(self.model.idp, 'has_soc', False):
            spindeg = 1
        elif hasattr(self.model, 'soc_param'):
            spindeg = 1
        else:
            spindeg = 2

        E_fermi = self.cal_E_fermi(eigs, total_nel, spindeg, wk,
                                   q_tol=q_tol, smearing_method=smearing_method, temp=temp)
        return data, E_fermi

    @classmethod
    def cal_E_fermi(cls, eigenvalues: np.ndarray, total_electrons: int, spindeg: int = 2, wk: np.ndarray = None,
                    q_tol: float = 1e-10, smearing_method: str = 'FD', temp: float = 300):
        nextafter = np.nextafter
        total_electrons = total_electrons / spindeg

        min_Ef, max_Ef = eigenvalues.min(), eigenvalues.max()
        kT = Boltzmann / eV2J * temp
        drange = kT * np.sqrt(-np.log(q_tol * 1e-2))
        min_Ef = min_Ef - drange
        max_Ef = max_Ef + drange

        Ef = (min_Ef + max_Ef) * 0.5

        if wk is None:
            wk = np.ones(eigenvalues.shape[0]) / eigenvalues.shape[0]

        while nextafter(min_Ef, max_Ef) < max_Ef:
            wk2 = wk.reshape(-1, 1)
            if smearing_method == 'FD':
                q_cal = (wk2 * cls.fermi_dirac_smearing(eigenvalues, kT=kT, mu=Ef)).sum()
            elif smearing_method == 'Gaussian':
                q_cal = (wk2 * cls.Gaussian_smearing(eigenvalues, sigma=kT, mu=Ef)).sum()
            else:
                raise ValueError(f'Unknown smearing method: {smearing_method}')

            if abs(q_cal - total_electrons) < q_tol:
                return Ef

            if q_cal >= total_electrons:
                max_Ef = Ef
            else:
                min_Ef = Ef
            Ef = (min_Ef + max_Ef) * 0.5

        return Ef

    @classmethod
    def fermi_dirac_smearing(cls, E, kT=0.025852, mu=0.0):
        x = (E - mu) / kT
        mask_min = x < -40.0
        mask_max = x > 40.0
        mask_in_limit = ~(mask_min | mask_max)
        out = np.zeros_like(x)
        out[mask_min] = 1.0
        out[mask_max] = 0.0
        out[mask_in_limit] = 1.0 / (np.expm1(x[mask_in_limit]) + 2.0)
        return out

    @classmethod
    def Gaussian_smearing(cls, E, sigma=0.025852, mu=0.0):
        from scipy.special import erfc
        x = (mu - E) / sigma
        return 0.5 * erfc(-1 * x)