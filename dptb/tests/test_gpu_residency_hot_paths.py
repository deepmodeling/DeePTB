from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _reference_mole_linear(layer, x, coeffs, sizes):
    torch = pytest.importorskip("torch")
    mixed_weights = torch.einsum("be,eoi->boi", coeffs, layer.weight_experts)
    if layer.num_shared_experts > 0:
        mixed_weights = mixed_weights + layer.weight_shared.sum(0).unsqueeze(0)

    mixed_bias = None
    if layer.bias_experts is not None:
        mixed_bias = torch.einsum("be,eo->bo", coeffs, layer.bias_experts)
        if layer.num_shared_experts > 0 and layer.bias_shared is not None:
            mixed_bias = mixed_bias + layer.bias_shared.sum(0).unsqueeze(0)

    out_parts = []
    for i, x_sys in enumerate(torch.split(x, tuple(int(v) for v in sizes), dim=0)):
        bias = None if mixed_bias is None else mixed_bias[i]
        out_parts.append(torch.nn.functional.linear(x_sys, mixed_weights[i], bias))
    return torch.cat(out_parts, dim=0)


def test_mole_globals_caches_split_sizes_and_linear_matches_reference():
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    torch.manual_seed(7)
    layer = MOLELinear(3, 4, num_experts=2, num_shared_experts=1, bias=True)
    coeffs = torch.tensor([[0.25, 0.75], [1.0, 0.0]], dtype=torch.float32)
    sizes = torch.tensor([2, 1], dtype=torch.long)
    globals_ = MOLEGlobals(coefficients=coeffs, sizes=sizes)

    assert globals_.split_sizes == (2, 1)

    x2 = torch.randn(3, 3)
    torch.testing.assert_close(layer(x2, globals_), _reference_mole_linear(layer, x2, coeffs, sizes))

    x3 = torch.randn(3, 2, 3)
    torch.testing.assert_close(layer(x3, globals_), _reference_mole_linear(layer, x3, coeffs, sizes))


def test_lem_active_edge_split_sizes_use_cpu_edge_slices():
    torch = pytest.importorskip("torch")
    from dptb.data import _keys
    from dptb.nnops.multi_trainer import MultiTrainer

    batch = {"__slices__": {_keys.EDGE_INDEX_KEY: torch.tensor([0, 3, 5, 9])}}
    active_edges = torch.tensor([0, 2, 4, 5, 8], dtype=torch.long)

    assert MultiTrainer._lem_active_edge_split_sizes(batch, active_edges) == (2, 1, 2)

    class BatchLike:
        __slices__ = {_keys.EDGE_INDEX_KEY: [0, 3, 5, 9]}

    assert MultiTrainer._lem_active_edge_split_sizes(BatchLike(), [0, 2, 4, 5, 8]) == (2, 1, 2)


def test_lem_split_sizes_are_reattached_as_cpu_tensor_after_to_dict():
    torch = pytest.importorskip("torch")
    from dptb.data import _keys
    from dptb.nnops.multi_trainer import MultiTrainer

    batch_dict = {}
    cpu_batch = {_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY: torch.tensor([2, 0, 3], dtype=torch.long)}

    MultiTrainer._attach_lem_cpu_split_sizes(batch_dict, cpu_batch)

    split_sizes = batch_dict[_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY]
    assert torch.is_tensor(split_sizes)
    assert split_sizes.device.type == "cpu"
    assert split_sizes.dtype == torch.long
    torch.testing.assert_close(split_sizes, torch.tensor([2, 0, 3], dtype=torch.long))


def test_lem_precompute_metadata_is_cleared_before_reuse():
    torch = pytest.importorskip("torch")
    from dptb.data import _keys
    from dptb.nnops.multi_trainer import MultiTrainer

    batch = {
        _keys.LEM_ACTIVE_EDGES_KEY: torch.tensor([0], dtype=torch.long),
        _keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY: torch.tensor([1], dtype=torch.long),
        _keys.LEM_CUTOFF_COEFFS_KEY: torch.tensor([1.0]),
    }

    MultiTrainer._clear_lem_precompute_metadata(batch)

    assert _keys.LEM_ACTIVE_EDGES_KEY not in batch
    assert _keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY not in batch
    assert _keys.LEM_CUTOFF_COEFFS_KEY not in batch


def test_multi_trainer_reattaches_lem_split_sizes_after_to_dict_source_guard():
    multi_trainer_text = _read_source(REPO_ROOT / "dptb" / "nnops" / "multi_trainer.py")

    assert "def _attach_lem_cpu_split_sizes" in multi_trainer_text
    assert "def _clear_lem_precompute_metadata" in multi_trainer_text
    assert "batch = self._clear_lem_precompute_metadata(batch)" in multi_trainer_text
    assert "batch[_keys.EDGE_VECTORS_KEY] = cutoff_data[_keys.EDGE_VECTORS_KEY]" in multi_trainer_text
    assert "batch[_keys.EDGE_LENGTH_KEY] = cutoff_data[_keys.EDGE_LENGTH_KEY]" in multi_trainer_text
    assert "batch_dict = self._attach_lem_cpu_split_sizes(batch_dict, batch)" in multi_trainer_text
    assert "rank0_batch_dict = self._attach_lem_cpu_split_sizes(rank0_batch_dict, batch)" in multi_trainer_text
    assert "rank0_ref_batch_dict = self._attach_lem_cpu_split_sizes(rank0_ref_batch_dict, ref_batch)" in multi_trainer_text
    assert "precomputed_cutoff_coeffs is not None and edge_length.requires_grad" in (
        _read_source(REPO_ROOT / "dptb" / "nn" / "embedding" / "lem_moe_v3.py")
    )
    assert "precomputed_cutoff_coeffs is not None and edge_length.requires_grad" in (
        _read_source(REPO_ROOT / "dptb" / "nn" / "embedding" / "lem_moe_v3_h0.py")
    )


def test_per_edge_cutoffs_match_old_loop_semantics():
    torch = pytest.importorskip("torch")
    from dptb.nn.cutoff import cosine_cutoff, polynomial_cutoff
    from dptb.nn.embedding.lem_moe_v3 import (
        _cosine_cutoff_per_edge,
        _polynomial_cutoff_per_edge,
    )

    edge_length = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float32)
    bond_r_max = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    valid = torch.tensor([True, True, False, True])

    polynomial = _polynomial_cutoff_per_edge(edge_length, bond_r_max, p=6.0)
    cosine = _cosine_cutoff_per_edge(edge_length, bond_r_max, r_start_cos_ratio=0.8)

    polynomial_ref = torch.stack(
        [
            polynomial_cutoff(edge_length[i : i + 1], bond_r_max[i : i + 1], p=6.0).flatten()[0]
            for i in range(edge_length.numel())
        ]
    )
    cosine_ref = torch.stack(
        [
            cosine_cutoff(
                edge_length[i : i + 1],
                bond_r_max[i : i + 1],
                r_start_cos_ratio=0.8,
            ).flatten()[0]
            for i in range(edge_length.numel())
        ]
    )

    torch.testing.assert_close(polynomial * valid, polynomial_ref * valid)
    torch.testing.assert_close(cosine * valid, cosine_ref * valid)


def test_current_gpu_hot_paths_do_not_extract_cuda_scalars_in_forward():
    tensor_product_text = _read_source(REPO_ROOT / "dptb" / "nn" / "tensor_product_moe_v3.py")
    atomic_data_text = _read_source(REPO_ROOT / "dptb" / "data" / "AtomicDataDict.py")
    loss_text = _read_source(REPO_ROOT / "dptb" / "nnops" / "loss.py")
    embedding_dir = REPO_ROOT / "dptb" / "nn" / "embedding"
    lem_text = _read_source(embedding_dir / "lem_moe_v3.py")
    h0_text = _read_source(embedding_dir / "lem_moe_v3_h0.py")
    h0_helpers_text = _read_source(embedding_dir / "lem_moe_v3_h0_helpers.py")

    assert "mole_globals.sizes.tolist()" not in tensor_product_text
    assert 'sizes = sizes.to("cpu")' not in tensor_product_text
    assert "sizes = sizes.to('cpu')" not in tensor_product_text
    assert 'sizes.to("cpu").tolist()' not in tensor_product_text
    assert "sizes.to('cpu').tolist()" not in tensor_product_text
    assert "sizes.sum().item()" not in tensor_product_text
    assert "torch.sum(torch.abs(data[_keys.CELL_KEY]))" not in atomic_data_text
    assert "batch.max() == 0" not in loss_text
    assert "batch.max() >= 1" not in loss_text
    assert "batch.max().item()" not in loss_text
    assert "torch.any(mask_" not in loss_text
    assert "batch.max().item()" not in lem_text
    assert "batch.max().item()" not in h0_text
    assert "if mask.any()" not in lem_text
    assert "if self_mask.any()" not in h0_helpers_text
