import pytest


@pytest.mark.parametrize("wigner_apply_mode", ["compact_blocks", "full_dense"])
@pytest.mark.parametrize("so2_fusion_mode", ["streamed_m_major_ref", "streamed_m_major_aggressive"])
@pytest.mark.parametrize(
    "rotate_in, rotate_out",
    [(True, True), (False, True), (True, False), (False, False)],
)
def test_so2_streamed_handles_out_lmax_gt_in_lmax(so2_fusion_mode, wigner_apply_mode, rotate_in, rotate_out):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear

    torch.manual_seed(20260423)
    dtype = torch.float64
    irreps_in = "1x0e + 2x1o + 1x2e"
    irreps_out = "1x0e + 1x1o + 1x2e + 1x3o"
    kwargs = dict(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        radial_emb=True,
        latent_dim=5,
        radial_channels=[7],
        num_experts=3,
        num_shared_experts=1,
        rotate_in=rotate_in,
        rotate_out=rotate_out,
        wigner_apply_mode=wigner_apply_mode,
    )
    staged = SO2_Linear(**kwargs, so2_fusion_mode="staged").to(dtype=dtype)
    streamed = SO2_Linear(**kwargs, so2_fusion_mode=so2_fusion_mode).to(dtype=dtype)
    streamed.load_state_dict(staged.state_dict(), strict=True)

    x0 = torch.randn(5, staged.irreps_in.dim, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    R0 = torch.randn(5, 3, dtype=dtype, requires_grad=True)
    R1 = R0.detach().clone().requires_grad_(True)
    lat0 = torch.randn(5, 5, dtype=dtype, requires_grad=True)
    lat1 = lat0.detach().clone().requires_grad_(True)
    coeffs = torch.tensor([[0.2, 0.3, 0.5], [0.7, 0.1, 0.2]], dtype=dtype)
    globals_ = MOLEGlobals(coefficients=coeffs, split_sizes=(2, 3))

    out0, _ = staged(x0, R0, globals_, lat0)
    out1, _ = streamed(x1, R1, globals_, lat1)
    torch.testing.assert_close(out1, out0, atol=1e-9, rtol=1e-9)

    probe = torch.randn_like(out0)
    (out0 * probe).sum().backward()
    (out1 * probe).sum().backward()

    torch.testing.assert_close(x1.grad, x0.grad, atol=1e-8, rtol=1e-8)
    if rotate_in or rotate_out:
        torch.testing.assert_close(R1.grad, R0.grad, atol=1e-8, rtol=1e-8)
    else:
        assert R0.grad is None
        assert R1.grad is None
    torch.testing.assert_close(lat1.grad, lat0.grad, atol=1e-8, rtol=1e-8)


def test_so2_fusion_mode_env_selects_aggressive(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    from dptb.nn.tensor_product_moe_v3 import SO2_Linear

    monkeypatch.setenv("DPTB_SO2_FUSION_MODE", "streamed_m_major_aggressive")
    layer = SO2_Linear(
        irreps_in="1x0e + 1x1o",
        irreps_out="1x0e + 1x1o",
        num_experts=2,
        num_shared_experts=0,
    )

    assert layer.so2_fusion_mode == "streamed_m_major_aggressive"


def test_so2_aggressive_cueq_indexed_linear_matches_staged_if_available():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    pytest.importorskip("cuequivariance")
    pytest.importorskip("cuequivariance_torch")
    if not torch.cuda.is_available():
        pytest.skip("SO2 aggressive cueq indexed-linear integration requires CUDA")

    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear

    torch.manual_seed(20260423)
    device = torch.device("cuda")
    dtype = torch.float32
    kwargs = dict(
        irreps_in="2x0e + 2x1o + 1x2e",
        irreps_out="1x0e + 2x1o + 2x2e + 1x3o",
        radial_emb=True,
        latent_dim=6,
        radial_channels=[8],
        num_experts=6,
        num_shared_experts=0,
        rotate_in=True,
        rotate_out=True,
        wigner_apply_mode="compact_blocks",
    )

    staged = SO2_Linear(
        **kwargs,
        so2_fusion_mode="staged",
        mole_linear_mode="split_loop",
    ).to(device=device, dtype=dtype)
    aggressive = SO2_Linear(
        **kwargs,
        so2_fusion_mode="streamed_m_major_aggressive",
        mole_linear_mode="cueq_indexed_linear",
    ).to(device=device, dtype=dtype)
    aggressive.load_state_dict(staged.state_dict(), strict=True)

    split_sizes = (3, 5, 4)
    n_edges = sum(split_sizes)
    coeffs = torch.rand(len(split_sizes), kwargs["num_experts"], device=device, dtype=dtype)
    coeffs = coeffs / coeffs.sum(dim=-1, keepdim=True)
    globals_ = MOLEGlobals(coefficients=coeffs, split_sizes=split_sizes)

    x0 = torch.randn(n_edges, staged.irreps_in.dim, device=device, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    R0 = torch.randn(n_edges, 3, device=device, dtype=dtype, requires_grad=True)
    R1 = R0.detach().clone().requires_grad_(True)
    lat0 = torch.randn(n_edges, kwargs["latent_dim"], device=device, dtype=dtype, requires_grad=True)
    lat1 = lat0.detach().clone().requires_grad_(True)

    out0, _ = staged(x0, R0, globals_, lat0)
    out1, _ = aggressive(x1, R1, globals_, lat1)
    torch.testing.assert_close(out1, out0, atol=3e-4, rtol=3e-4)

    probe = torch.randn_like(out0)
    (out0 * probe).mean().backward()
    (out1 * probe).mean().backward()
    torch.testing.assert_close(x1.grad, x0.grad, atol=4e-4, rtol=4e-4)
    torch.testing.assert_close(R1.grad, R0.grad, atol=4e-4, rtol=4e-4)
    torch.testing.assert_close(lat1.grad, lat0.grad, atol=4e-4, rtol=4e-4)
