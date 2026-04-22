import pytest

pytest.importorskip("torch")

from dptb.plugins.monitor import CUDAMemoryMonitor


class DummyTrainer:
    def __init__(self, num_experts=2):
        self.stats = {}
        self.num_experts = num_experts


def test_cuda_memory_monitor_tracks_global_and_expert_peak_fields():
    trainer = DummyTrainer(num_experts=2)
    monitor = CUDAMemoryMonitor(interval=[(1, "iteration"), (1, "epoch")])
    monitor.register(trainer)

    monitor.iteration(
        cuda_peak_allocated_mb=100.0,
        cuda_peak_reserved_mb=120.0,
        expert_0_cuda_peak_allocated_mb=90.0,
        expert_1_cuda_peak_allocated_mb=80.0,
        train_loss=1.5,
    )
    monitor.iteration(
        cuda_peak_allocated_mb=85.0,
        cuda_peak_reserved_mb=130.0,
        expert_0_cuda_peak_allocated_mb=95.0,
    )
    monitor.epoch()

    assert trainer.stats["cuda_peak_allocated_mb"]["last"] == 85.0
    assert trainer.stats["cuda_peak_allocated_mb"]["max"] == 100.0
    assert trainer.stats["cuda_peak_allocated_mb"]["epoch_max"] == 100.0
    assert trainer.stats["cuda_peak_reserved_mb"]["max"] == 130.0
    assert trainer.stats["expert_0_cuda_peak_allocated_mb"]["max"] == 95.0
    assert trainer.stats["expert_1_cuda_peak_allocated_mb"]["max"] == 80.0
    assert "train_loss" not in trainer.stats
    assert trainer.stats["cuda_peak_allocated_mb"]["log_unit"] == "MB"

    monitor.iteration(cuda_peak_allocated_mb=70.0)
    monitor.epoch()

    assert trainer.stats["cuda_peak_allocated_mb"]["epoch_max"] == 70.0
    assert trainer.stats["cuda_peak_allocated_mb"]["max"] == 100.0
