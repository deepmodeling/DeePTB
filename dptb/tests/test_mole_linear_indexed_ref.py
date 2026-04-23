import pytest


@pytest.fixture(autouse=True)
def _clear_mole_linear_mode_env(monkeypatch):
    monkeypatch.delenv("DPTB_MOLE_LINEAR_MODE", raising=False)


def _make_globals(torch, *, device, dtype, sizes=(3, 5, 2, 7), num_experts=8):
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals

    sizes = torch.tensor(sizes, device=device)
    coeffs = torch.rand(int(sizes.numel()), num_experts, device=device, dtype=dtype)
    coeffs = coeffs / coeffs.sum(dim=-1, keepdim=True)
    return MOLEGlobals(coefficients=coeffs, sizes=sizes), int(sizes.sum().item())


def _assert_mole_modes_match(torch, *, shape, bias, num_shared_experts, device, dtype, sizes=(3, 5, 2, 7)):
    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    num_experts = 8
    in_features = shape[-1]
    out_features = 13
    globals_, _ = _make_globals(torch, device=device, dtype=dtype, sizes=sizes, num_experts=num_experts)

    base = MOLELinear(
        in_features,
        out_features,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        bias=bias,
        mole_linear_mode="split_loop",
    ).to(device=device, dtype=dtype)
    indexed = MOLELinear(
        in_features,
        out_features,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        bias=bias,
        mole_linear_mode="indexed_ref",
    ).to(device=device, dtype=dtype)
    indexed.load_state_dict(base.state_dict(), strict=True)

    x0 = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)

    y0 = base(x0, globals_)
    y1 = indexed(x1, globals_)
    torch.testing.assert_close(y1, y0, atol=1e-10, rtol=1e-10)

    probe = torch.randn_like(y0)
    (y0 * probe).sum().backward()
    (y1 * probe).sum().backward()

    torch.testing.assert_close(x1.grad, x0.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        indexed.weight_experts.grad,
        base.weight_experts.grad,
        atol=1e-10,
        rtol=1e-10,
    )
    if bias:
        torch.testing.assert_close(
            indexed.bias_experts.grad,
            base.bias_experts.grad,
            atol=1e-10,
            rtol=1e-10,
        )
    if num_shared_experts > 0:
        torch.testing.assert_close(
            indexed.weight_shared.grad,
            base.weight_shared.grad,
            atol=1e-10,
            rtol=1e-10,
        )
        if bias:
            torch.testing.assert_close(
                indexed.bias_shared.grad,
                base.bias_shared.grad,
                atol=1e-10,
                rtol=1e-10,
            )


def test_mole_linear_indexed_ref_matches_split_loop_forward_and_grad():
    torch = pytest.importorskip("torch")

    torch.manual_seed(20260423)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    _, n_edges = _make_globals(torch, device=device, dtype=dtype)

    _assert_mole_modes_match(
        torch,
        shape=(n_edges, 11),
        bias=True,
        num_shared_experts=2,
        device=device,
        dtype=dtype,
    )
    _assert_mole_modes_match(
        torch,
        shape=(n_edges, 2, 11),
        bias=True,
        num_shared_experts=2,
        device=device,
        dtype=dtype,
    )


def test_mole_linear_indexed_ref_matches_split_loop_without_bias_or_shared_experts():
    torch = pytest.importorskip("torch")

    torch.manual_seed(20260424)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    _, n_edges = _make_globals(torch, device=device, dtype=dtype, sizes=(1, 4, 6))

    _assert_mole_modes_match(
        torch,
        shape=(n_edges, 2, 7),
        bias=False,
        num_shared_experts=0,
        device=device,
        dtype=dtype,
        sizes=(1, 4, 6),
    )
    _assert_mole_modes_match(
        torch,
        shape=(n_edges, 7),
        bias=True,
        num_shared_experts=0,
        device=device,
        dtype=dtype,
        sizes=(1, 4, 6),
    )
    _assert_mole_modes_match(
        torch,
        shape=(n_edges, 2, 7),
        bias=False,
        num_shared_experts=2,
        device=device,
        dtype=dtype,
        sizes=(1, 4, 6),
    )


def test_mole_linear_fallback_average_ignores_indexed_mode():
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    torch.manual_seed(20260425)
    dtype = torch.float64
    base = MOLELinear(5, 3, num_experts=4, num_shared_experts=1, bias=True, mole_linear_mode="split_loop").to(dtype=dtype)
    indexed = MOLELinear(5, 3, num_experts=4, num_shared_experts=1, bias=True, mole_linear_mode="indexed_ref").to(dtype=dtype)
    indexed.load_state_dict(base.state_dict(), strict=True)

    x = torch.randn(6, 2, 5, dtype=dtype)
    torch.testing.assert_close(indexed(x, None), base(x, None), atol=1e-10, rtol=1e-10)


def test_mole_linear_indexed_ref_matches_coefficients_grad():
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    torch.manual_seed(20260426)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    sizes = (2, 3, 4)
    num_graphs = len(sizes)
    num_experts = 5
    n_edges = sum(sizes)

    coeffs = torch.rand(num_graphs, num_experts, device=device, dtype=dtype)
    coeffs = (coeffs / coeffs.sum(dim=-1, keepdim=True)).detach()
    coeffs0 = coeffs.clone().requires_grad_(True)
    coeffs1 = coeffs.clone().requires_grad_(True)
    globals0 = MOLEGlobals(coefficients=coeffs0, split_sizes=sizes)
    globals1 = MOLEGlobals(coefficients=coeffs1, split_sizes=sizes)

    base = MOLELinear(6, 8, num_experts=num_experts, num_shared_experts=1, bias=True, mole_linear_mode="split_loop").to(device=device, dtype=dtype)
    indexed = MOLELinear(6, 8, num_experts=num_experts, num_shared_experts=1, bias=True, mole_linear_mode="indexed_ref").to(device=device, dtype=dtype)
    indexed.load_state_dict(base.state_dict(), strict=True)

    x0 = torch.randn(n_edges, 2, 6, device=device, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    probe = torch.randn(n_edges, 2, 8, device=device, dtype=dtype)

    (base(x0, globals0) * probe).sum().backward()
    (indexed(x1, globals1) * probe).sum().backward()

    torch.testing.assert_close(coeffs1.grad, coeffs0.grad, atol=1e-10, rtol=1e-10)


def test_mole_globals_explicit_split_sizes_take_precedence():
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    torch.manual_seed(20260427)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    num_experts = 5
    split_sizes = (2, 4, 3)
    misleading_sizes = torch.tensor((1, 1, 1), device=device)

    coeffs = torch.rand(len(split_sizes), num_experts, device=device, dtype=dtype)
    coeffs = coeffs / coeffs.sum(dim=-1, keepdim=True)
    globals_ = MOLEGlobals(coefficients=coeffs, sizes=misleading_sizes, split_sizes=split_sizes)

    base = MOLELinear(6, 8, num_experts=num_experts, num_shared_experts=1, bias=True, mole_linear_mode="split_loop").to(device=device, dtype=dtype)
    indexed = MOLELinear(6, 8, num_experts=num_experts, num_shared_experts=1, bias=True, mole_linear_mode="indexed_ref").to(device=device, dtype=dtype)
    indexed.load_state_dict(base.state_dict(), strict=True)

    x = torch.randn(sum(split_sizes), 6, device=device, dtype=dtype)
    torch.testing.assert_close(indexed(x, globals_), base(x, globals_), atol=1e-10, rtol=1e-10)


def test_mole_linear_env_selects_indexed_ref(monkeypatch):
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    monkeypatch.setenv("DPTB_MOLE_LINEAR_MODE", "indexed_ref")
    layer = MOLELinear(4, 4)
    assert layer.mole_linear_mode == "indexed_ref"


def test_mole_linear_invalid_mode_rejected():
    pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    with pytest.raises(ValueError, match="mole_linear_mode"):
        MOLELinear(4, 4, mole_linear_mode="bad")


def test_mole_linear_cueq_indexed_smoke_if_available():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cuequivariance")
    pytest.importorskip("cuequivariance_torch")
    if not torch.cuda.is_available():
        pytest.skip("cueq indexed linear smoke requires CUDA")

    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    torch.manual_seed(20260423)
    device = torch.device("cuda")
    dtype = torch.float32
    sizes = (4, 6, 5)
    num_experts = 6
    in_features = 7
    out_features = 9
    globals_, n_edges = _make_globals(
        torch,
        device=device,
        dtype=dtype,
        sizes=sizes,
        num_experts=num_experts,
    )

    base = MOLELinear(
        in_features,
        out_features,
        num_experts=num_experts,
        num_shared_experts=1,
        bias=True,
        mole_linear_mode="split_loop",
    ).to(device=device, dtype=dtype)
    cueq = MOLELinear(
        in_features,
        out_features,
        num_experts=num_experts,
        num_shared_experts=1,
        bias=True,
        mole_linear_mode="cueq_indexed_linear",
    ).to(device=device, dtype=dtype)
    cueq.load_state_dict(base.state_dict(), strict=True)

    x0 = torch.randn(n_edges, 2, in_features, device=device, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    y0 = base(x0, globals_)
    y1 = cueq(x1, globals_)
    torch.testing.assert_close(y1, y0, atol=2e-4, rtol=2e-4)

    probe = torch.randn_like(y0)
    (y0 * probe).mean().backward()
    (y1 * probe).mean().backward()
    torch.testing.assert_close(x1.grad, x0.grad, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(cueq.weight_experts.grad, base.weight_experts.grad, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(cueq.bias_experts.grad, base.bias_experts.grad, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(cueq.weight_shared.grad, base.weight_shared.grad, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(cueq.bias_shared.grad, base.bias_shared.grad, atol=2e-4, rtol=2e-4)
    assert cueq._cueq_weight_order == "io_scaled"


def test_mole_linear_cueq_env_smoke_if_available(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("cuequivariance")
    pytest.importorskip("cuequivariance_torch")
    if not torch.cuda.is_available():
        pytest.skip("cueq indexed linear smoke requires CUDA")

    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    monkeypatch.setenv("DPTB_MOLE_LINEAR_MODE", "cueq_indexed_linear")
    assert MOLELinear(3, 3).mole_linear_mode == "cueq_indexed_linear"


def test_mole_linear_cueq_rejects_amp_dtype_if_available():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cuequivariance")
    pytest.importorskip("cuequivariance_torch")
    if not torch.cuda.is_available():
        pytest.skip("cueq indexed linear smoke requires CUDA")

    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    device = torch.device("cuda")
    dtype = torch.float16
    coeffs = torch.rand(2, 4, device=device, dtype=dtype)
    coeffs = coeffs / coeffs.sum(dim=-1, keepdim=True)
    globals_ = MOLEGlobals(coefficients=coeffs, split_sizes=(2, 3))
    layer = MOLELinear(
        3,
        5,
        num_experts=4,
        num_shared_experts=0,
        bias=False,
        mole_linear_mode="cueq_indexed_linear",
    ).to(device=device, dtype=dtype)
    x = torch.randn(5, 3, device=device, dtype=dtype)

    with pytest.raises(RuntimeError, match="float32/float64"):
        layer(x, globals_)
