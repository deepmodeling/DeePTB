import pytest


def _clone_router(torch, base, cls, *, full_expert_fast_path):
    clone = cls(
        in_features=base.net[0].in_features,
        num_experts=base.num_experts,
        top_k=base.top_k,
        aux_loss_free=base.aux_loss_free,
        bias_update_speed=base.bias_update_speed,
        full_expert_fast_path=full_expert_fast_path,
    )
    clone.load_state_dict(base.state_dict(), strict=True)
    return clone


def test_full_expert_fast_path_matches_all_expert_topk_and_skips_load_updates():
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLERouterV3

    torch.manual_seed(20260423)
    num_experts = 6
    base = MOLERouterV3(
        in_features=9,
        num_experts=num_experts,
        top_k=num_experts,
        aux_loss_free=True,
        bias_update_speed=0.005,
        full_expert_fast_path=False,
    )
    fast = _clone_router(torch, base, MOLERouterV3, full_expert_fast_path=True)

    base.train()
    fast.train()
    global_features = torch.randn(5, 9, requires_grad=True)
    sizes = torch.tensor([2, 4, 3, 5, 7], dtype=torch.float32)

    coeff_slow, monitor_slow, cv_slow = base(global_features, sizes=sizes)
    coeff_fast, monitor_fast, cv_fast = fast(global_features, sizes=sizes)

    torch.testing.assert_close(coeff_fast, coeff_slow, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(monitor_fast, monitor_slow, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(cv_fast, cv_slow, atol=1e-7, rtol=1e-7)

    torch.testing.assert_close(fast.expert_bias, torch.zeros_like(fast.expert_bias))
    torch.testing.assert_close(fast.ema_load, torch.ones_like(fast.ema_load))
    assert not torch.equal(base.ema_load, fast.ema_load)


def test_full_expert_fast_path_matches_dense_none_topk():
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLERouterV3

    torch.manual_seed(20260424)
    base = MOLERouterV3(
        in_features=7,
        num_experts=5,
        top_k=None,
        aux_loss_free=True,
        full_expert_fast_path=False,
    )
    fast = _clone_router(torch, base, MOLERouterV3, full_expert_fast_path=True)

    global_features = torch.randn(4, 7, requires_grad=True)
    coeff_base, monitor_base, cv_base = base(global_features)
    coeff_fast, monitor_fast, cv_fast = fast(global_features)

    torch.testing.assert_close(coeff_fast, coeff_base, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(monitor_fast, monitor_base, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(cv_fast, cv_base, atol=1e-7, rtol=1e-7)


def test_full_expert_fast_path_does_not_change_true_topk_behavior():
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLERouterV3

    torch.manual_seed(20260425)
    base = MOLERouterV3(
        in_features=8,
        num_experts=6,
        top_k=2,
        aux_loss_free=True,
        bias_update_speed=0.005,
        full_expert_fast_path=False,
    )
    fast_flag = _clone_router(torch, base, MOLERouterV3, full_expert_fast_path=True)

    base.train()
    fast_flag.train()
    global_features = torch.randn(6, 8, requires_grad=True)
    coeff_base, monitor_base, cv_base = base(global_features)
    coeff_fast, monitor_fast, cv_fast = fast_flag(global_features)

    torch.testing.assert_close(coeff_fast, coeff_base, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(monitor_fast, monitor_base, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(cv_fast, cv_base, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(fast_flag.expert_bias, base.expert_bias, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(fast_flag.ema_load, base.ema_load, atol=1e-7, rtol=1e-7)
