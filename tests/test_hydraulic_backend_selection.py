import pytest

from tests.comprehensive_hil_testing.hydraulic_backends import (
    SimpleTankBackend,
    build_single_pool_backend,
)


class _DummyTank:
    def __init__(self, *, area, dt, delay_steps, initial_level, initial_flow):
        self.level = float(initial_level)
        self.area = float(area)

    def step(self, q_in_command, q_out):
        self.level += float(q_in_command - q_out) / max(self.area, 1.0)
        return self.level

    def get_level(self):
        return self.level

    def get_volume(self):
        return self.level * self.area


def _build_backend(backend: str):
    return build_single_pool_backend(
        backend=backend,
        area=10_000.0,
        slope=1e-4,
        time_step=60.0,
        initial_level=3.0,
        initial_flow=5.0,
        manning_n=0.02,
        tank_cls=_DummyTank,
        fidelity_cls=None,
        geometry_cls=None,
        segmented_model_cls=None,
        segmented_config_cls=None,
        segmented_boundary_type_cls=None,
        delay_steps=1,
    )


def test_explicit_single_channel_backend_does_not_fallback_to_tank():
    with pytest.raises(RuntimeError, match="single_channel backend unavailable"):
        _build_backend("single_channel")


def test_explicit_segmented_hf_backend_does_not_fallback_to_tank():
    with pytest.raises(RuntimeError, match="segmented_hf backend unavailable"):
        _build_backend("segmented_hf")


def test_auto_backend_can_fallback_to_tank():
    backend = _build_backend("auto")

    assert isinstance(backend, SimpleTankBackend)
    assert backend.name == "tank"
