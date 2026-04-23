from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "dptb" / "nn" / "tensor_product_moe_v3.py"
LEM_SOURCE_PATH = REPO_ROOT / "dptb" / "nn" / "embedding" / "lem_moe_v3.py"
ARGCHECK_SOURCE_PATH = REPO_ROOT / "dptb" / "utils" / "argcheck.py"


def _read_source() -> str:
    return SOURCE_PATH.read_text(encoding="utf-8", errors="ignore")


def test_tensor_product_source_exposes_compact_wigner_dual_path():
    source = _read_source()

    assert "class SO2WignerBlocks" in source
    assert "def batch_wigner_D_blocks" in source
    assert "wigner_apply_mode" in source
    assert "so2_fusion_mode" in source
    assert '"streamed_m_major_ref"' in source
    assert '"full_dense"' in source
    assert '"compact_blocks"' in source


def test_compact_wigner_path_is_default_with_dense_fallback():
    source = _read_source()

    assert 'wigner_apply_mode: str = "compact_blocks"' in source
    assert 'self.wigner_apply_mode = _normalize_wigner_apply_mode(wigner_apply_mode)' in source
    assert 'if wigner_apply_mode == "compact_blocks":' in source
    assert "return batch_wigner_D(l_max, alpha, beta, gamma, _Jd)" in source


def test_compact_wigner_config_is_threaded_to_lem():
    lem_source = LEM_SOURCE_PATH.read_text(encoding="utf-8", errors="ignore")
    argcheck_source = ARGCHECK_SOURCE_PATH.read_text(encoding="utf-8", errors="ignore")

    assert 'so2_wigner_apply_mode: str = "compact_blocks"' in lem_source
    assert 'so2_fusion_mode: str = "staged"' in lem_source
    assert "wigner_apply_mode=so2_wigner_apply_mode" in lem_source
    assert "so2_fusion_mode=so2_fusion_mode" in lem_source
    assert 'Argument("so2_wigner_apply_mode", str' in argcheck_source
    assert 'Argument("so2_fusion_mode", str' in argcheck_source
    assert 'default="compact_blocks"' in argcheck_source


def test_compact_wigner_blocks_match_full_dense_slices():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    from dptb.nn.tensor_product_moe_v3 import (
        _Jd,
        batch_wigner_D,
        batch_wigner_D_blocks,
    )

    dtype = torch.float64
    alpha = torch.tensor([0.1, -0.2, 0.7], dtype=dtype)
    beta = torch.tensor([0.3, 0.5, -0.4], dtype=dtype)
    gamma = torch.zeros_like(alpha)
    l_max = 4

    full = batch_wigner_D(l_max, alpha, beta, gamma, _Jd)
    compact = batch_wigner_D_blocks(l_max, alpha, beta, gamma, _Jd)

    for l in range(l_max + 1):
        dim = 2 * l + 1
        start = l * l
        torch.testing.assert_close(compact.block(l), full[:, start : start + dim, start : start + dim])


def test_so2_linear_default_mode_is_compact_blocks():
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    from dptb.nn.tensor_product_moe_v3 import SO2_Linear

    layer = SO2_Linear("1x0e + 1x1e", "1x0e + 1x1e")

    assert layer.wigner_apply_mode == "compact_blocks"


def test_so2_linear_compact_mode_matches_full_dense_forward_and_grad():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear

    torch.manual_seed(17)
    irreps = "2x0e + 2x1e + 1x2e"
    dense_layer = SO2_Linear(
        irreps,
        irreps,
        num_experts=3,
        num_shared_experts=1,
        wigner_apply_mode="full_dense",
    )
    compact_layer = SO2_Linear(
        irreps,
        irreps,
        num_experts=3,
        num_shared_experts=1,
        wigner_apply_mode="compact_blocks",
    )
    compact_layer.load_state_dict(dense_layer.state_dict())

    x_dense = torch.randn(5, dense_layer.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    x_compact = x_dense.detach().clone().requires_grad_(True)
    R = torch.randn(5, 3, dtype=torch.float64)
    coeffs = torch.tensor(
        [[0.1, 0.3, 0.6], [0.2, 0.2, 0.6]],
        dtype=torch.float64,
    )
    mole_globals = MOLEGlobals(coefficients=coeffs, sizes=torch.tensor([2, 3]))

    dense_layer = dense_layer.to(dtype=torch.float64)
    compact_layer = compact_layer.to(dtype=torch.float64)

    dense_out, dense_wigner = dense_layer(x_dense, R, mole_globals)
    compact_out, compact_wigner = compact_layer(x_compact, R, mole_globals)

    torch.testing.assert_close(compact_out, dense_out, atol=1e-9, rtol=1e-9)
    assert hasattr(compact_wigner, "block")
    assert not hasattr(dense_wigner, "block")

    dense_out.square().sum().backward()
    compact_out.square().sum().backward()
    torch.testing.assert_close(x_compact.grad, x_dense.grad, atol=1e-9, rtol=1e-9)


def test_so2_linear_streamed_ref_matches_staged_forward_and_grad():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear

    torch.manual_seed(37)
    irreps_in = "1x0e + 2x1o + 1x2e"
    irreps_out = "1x0e + 1x1o + 2x2e"
    kwargs = dict(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        radial_emb=True,
        latent_dim=5,
        radial_channels=[7],
        num_experts=3,
        num_shared_experts=1,
        wigner_apply_mode="compact_blocks",
        rotate_in=True,
        rotate_out=True,
    )
    staged = SO2_Linear(**kwargs, so2_fusion_mode="staged").to(dtype=torch.float64)
    streamed = SO2_Linear(**kwargs, so2_fusion_mode="streamed_m_major_ref").to(dtype=torch.float64)
    streamed.load_state_dict(staged.state_dict())

    x_staged = torch.randn(5, staged.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    x_streamed = x_staged.detach().clone().requires_grad_(True)
    R_staged = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    R_streamed = R_staged.detach().clone().requires_grad_(True)
    latents_staged = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
    latents_streamed = latents_staged.detach().clone().requires_grad_(True)
    coeffs = torch.tensor(
        [[0.2, 0.3, 0.5], [0.7, 0.1, 0.2]],
        dtype=torch.float64,
    )
    mole_globals = MOLEGlobals(coefficients=coeffs, split_sizes=(2, 3))

    out_staged, wigner_staged = staged(x_staged, R_staged, mole_globals, latents_staged)
    out_streamed, wigner_streamed = streamed(x_streamed, R_streamed, mole_globals, latents_streamed)
    torch.testing.assert_close(out_streamed, out_staged, atol=1e-9, rtol=1e-9)
    assert hasattr(wigner_staged, "block")
    assert hasattr(wigner_streamed, "block")

    probe = torch.randn_like(out_staged)
    (out_staged * probe).sum().backward()
    (out_streamed * probe).sum().backward()

    torch.testing.assert_close(x_streamed.grad, x_staged.grad, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(R_streamed.grad, R_staged.grad, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(latents_streamed.grad, latents_staged.grad, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(
        streamed.fc_m0.weight_experts.grad,
        staged.fc_m0.weight_experts.grad,
        atol=1e-8,
        rtol=1e-8,
    )


def test_so2_linear_streamed_ref_matches_staged_rotate_flags_front_false():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear

    torch.manual_seed(41)
    irreps_in = "2x0e + 2x1o + 2x2e"
    irreps_out = "1x0e + 1x1o + 1x2e"
    coeffs = torch.tensor(
        [[0.4, 0.4, 0.2], [0.1, 0.3, 0.6]],
        dtype=torch.float64,
    )
    mole_globals = MOLEGlobals(coefficients=coeffs, split_sizes=(3, 2))

    for rotate_in, rotate_out in [(True, True), (False, True), (True, False), (False, False)]:
        kwargs = dict(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            radial_emb=True,
            latent_dim=4,
            radial_channels=[6],
            num_experts=3,
            num_shared_experts=1,
            wigner_apply_mode="compact_blocks",
            rotate_in=rotate_in,
            rotate_out=rotate_out,
        )
        staged = SO2_Linear(**kwargs, so2_fusion_mode="staged").to(dtype=torch.float64)
        streamed = SO2_Linear(**kwargs, so2_fusion_mode="streamed_m_major_ref").to(dtype=torch.float64)
        streamed.load_state_dict(staged.state_dict())

        x = torch.randn(5, staged.irreps_in.dim, dtype=torch.float64)
        R = torch.randn(5, 3, dtype=torch.float64)
        latents = torch.randn(5, 4, dtype=torch.float64)

        out_staged, _ = staged(x, R, mole_globals, latents)
        out_streamed, _ = streamed(x, R, mole_globals, latents)
        torch.testing.assert_close(out_streamed, out_staged, atol=1e-9, rtol=1e-9)


def test_so2_linear_streamed_ref_matches_staged_full_dense_wigner():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear

    torch.manual_seed(43)
    irreps = "1x0e + 1x1o + 1x2e"
    kwargs = dict(
        irreps_in=irreps,
        irreps_out=irreps,
        num_experts=3,
        num_shared_experts=1,
        wigner_apply_mode="full_dense",
        rotate_in=True,
        rotate_out=True,
    )
    staged = SO2_Linear(**kwargs, so2_fusion_mode="staged").to(dtype=torch.float64)
    streamed = SO2_Linear(**kwargs, so2_fusion_mode="streamed_m_major_ref").to(dtype=torch.float64)
    streamed.load_state_dict(staged.state_dict())

    x = torch.randn(4, staged.irreps_in.dim, dtype=torch.float64)
    R = torch.randn(4, 3, dtype=torch.float64)
    coeffs = torch.tensor(
        [[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]],
        dtype=torch.float64,
    )
    mole_globals = MOLEGlobals(coefficients=coeffs, split_sizes=(1, 3))

    out_staged, dense_wigner = staged(x, R, mole_globals)
    out_streamed, streamed_wigner = streamed(x, R, mole_globals)
    torch.testing.assert_close(out_streamed, out_staged, atol=1e-9, rtol=1e-9)
    assert not hasattr(dense_wigner, "block")
    assert not hasattr(streamed_wigner, "block")
