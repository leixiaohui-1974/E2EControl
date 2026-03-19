import numpy as np

from hydroe2e.phase5.distributed_sil.models.segmented_high_fidelity import (
    BoundaryType,
    SaintVenantConfig,
    SegmentedHighFidelityModel,
)


def _rectangular_model() -> SegmentedHighFidelityModel:
    model = SegmentedHighFidelityModel(
        segment_id="CORE_REGRESSION",
        length=1000.0,
        config=SaintVenantConfig(num_nodes=5, dt=1.0),
    )
    for section in model.sections:
        section.bottom_width = 5.0
        section.side_slope = 0.0
        section.manning_n = 0.025
    model.reset(initial_level=2.0, initial_flow=3.0)
    return model


def test_segmented_high_fidelity_mass_balance_uses_boundary_face_fluxes():
    model = _rectangular_model()
    model.set_boundary_conditions(
        upstream={"type": BoundaryType.DIRICHLET_FLOW, "value": 5.0},
        downstream={"type": BoundaryType.DIRICHLET_FLOW, "value": 1.0},
    )

    result = model.step()["mass_balance"]

    assert result["inflow"] == 5.0
    assert result["outflow"] == 1.0
    assert abs(result["storage_change"] - 4.0) <= 1e-6
    assert abs(result["balance_error"]) <= 1e-6


def test_segmented_high_fidelity_boundary_application_does_not_copy_neighbor_state():
    model = _rectangular_model()
    model.h[:] = np.array([2.0, 2.2, 2.4, 2.6, 2.8], dtype=float)
    model.Q[:] = np.array([3.0, 3.2, 3.4, 3.6, 3.8], dtype=float)
    model.set_boundary_conditions(
        upstream={"type": BoundaryType.DIRICHLET_FLOW, "value": 6.0},
        downstream={"type": BoundaryType.DIRICHLET_LEVEL, "value": 1.5},
    )

    model._apply_boundary_conditions()

    assert model.Q[0] == 6.0
    assert model.h[-1] == 1.5
    assert model.h[0] == 2.0
    assert model.Q[-1] == 3.8
