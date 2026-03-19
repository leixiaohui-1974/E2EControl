"""Adapters for strict revalidation hydraulic backends.

The strict testers historically used ``CanalPoolSimulator`` directly, which
reduces the plant to a storage tank with delayed inflow.  This module adds a
small compatibility layer so strict revalidation can exercise a higher-fidelity
plant without rewriting every test metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _manning_flow(width: float, depth: float, slope: float, manning_n: float, side_slope: float) -> float:
    wetted_area = width * depth + side_slope * depth**2
    wetted_perimeter = width + 2.0 * depth * np.sqrt(1.0 + side_slope**2)
    hydraulic_radius = wetted_area / max(wetted_perimeter, 1e-9)
    return (wetted_area / max(manning_n, 1e-9)) * (hydraulic_radius ** (2.0 / 3.0)) * np.sqrt(max(slope, 1e-9))


def _trapezoid_area(width: float, depth: float, side_slope: float) -> float:
    return width * depth + side_slope * depth**2


def _solve_bottom_width(
    *,
    depth: float,
    target_flow: float,
    slope: float,
    manning_n: float,
    side_slope: float,
) -> float:
    if target_flow <= 0.0:
        return 5.0

    lo = 1e-6
    hi = 500.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _manning_flow(mid, depth, slope, manning_n, side_slope) < target_flow:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _select_trapezoid_shape(
    *,
    depth: float,
    target_flow: float,
    slope: float,
    manning_n: float,
) -> tuple[float, float]:
    """Choose a feasible trapezoid shape for the requested depth/flow pair.

    Deep, low-flow cases can be infeasible for a wide trapezoid with side
    slope 2.0 even when the bottom width tends to zero. In that regime we
    reduce the side slope until the Manning conveyance can represent the
    requested operating point instead of forcing the backend into a strongly
    draining initial transient.
    """

    if target_flow <= 0.0:
        return 5.0, 2.0

    for side_slope in (2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.0):
        min_supported_flow = _manning_flow(
            1e-6,
            depth,
            slope,
            manning_n,
            side_slope,
        )
        if min_supported_flow <= target_flow:
            bottom_width = _solve_bottom_width(
                depth=depth,
                target_flow=target_flow,
                slope=slope,
                manning_n=manning_n,
                side_slope=side_slope,
            )
            return bottom_width, side_slope

    return 5.0, 0.0


def _fidelity_geometry(
    area: float,
    slope: float,
    initial_level: float,
    initial_flow: float,
    manning_n: float,
    geometry_cls: type[Any],
) -> Any:
    """Map the scenario's effective storage area to a self-consistent channel geometry.

    The strict scenarios describe an effective surface area for the dominant
    water body.  For the fidelity plant we keep that surface area approximately
    consistent while also making the initial depth/flow pair roughly satisfy a
    Manning steady state, so the backend does not introduce artificial drift at
    t=0 simply because the geometry was inconsistent.
    """

    depth = max(float(initial_level), 0.5)
    width, side_slope = _select_trapezoid_shape(
        depth=depth,
        target_flow=max(float(initial_flow), 0.0),
        slope=max(float(slope), 1e-5),
        manning_n=max(float(manning_n), 1e-4),
    )
    top_width = width + 2.0 * side_slope * depth
    length = max(100.0, float(area) / max(top_width, 1.0))
    return geometry_cls(
        length=length,
        N=20,
        width=width,
        slope=max(float(slope), 1e-5),
        side_slope=side_slope,
    )


@dataclass
class SimpleTankBackend:
    simulator: Any
    name: str = "tank"

    def step(self, q_in_command: float, q_out: float) -> float:
        return float(self.simulator.step(q_in_command=q_in_command, q_out=q_out))

    def get_level(self) -> float:
        return float(self.simulator.get_level())

    def get_volume(self) -> float:
        return float(self.simulator.get_volume())


class SingleChannelFidelityBackend:
    """Compatibility wrapper around ``SingleChannelFidelity``.

    The fidelity model evolves on a shorter internal time step.  The wrapper
    performs sub-stepping so the external strict tests can continue using the
    scenario time step unchanged.
    """

    def __init__(
        self,
        *,
        area: float,
        slope: float,
        time_step: float,
        initial_level: float,
        initial_flow: float,
        manning_n: float,
        geometry_cls: type[Any],
        fidelity_cls: type[Any],
    ) -> None:
        geometry = _fidelity_geometry(area, slope, initial_level, initial_flow, manning_n, geometry_cls)
        self.external_dt = float(time_step)
        self.sub_steps = max(1, int(np.ceil(self.external_dt / 300.0)))
        self.internal_dt = self.external_dt / self.sub_steps
        self.model = fidelity_cls(geometry=geometry, dt=self.internal_dt)
        self.model.state.Z[:] = float(initial_level)
        self.model.state.Q[:] = float(initial_flow)
        self.model.state.n_roughness[:] = float(manning_n)
        self.name = "single_channel_fidelity"

    def step(self, q_in_command: float, q_out: float) -> float:
        for _ in range(self.sub_steps):
            self.model.step(u_in=float(q_in_command), u_out=float(q_out))
        return self.get_level()

    def get_level(self) -> float:
        return float(np.mean(self.model.state.Z))

    def get_volume(self) -> float:
        dx = self.model.geom.segment_length()
        area_profile = [self.model._compute_area(depth) for depth in self.model.state.Z]
        return float(np.sum(area_profile) * dx)


class SegmentedHighFidelityBackend:
    """Compatibility wrapper around ``SegmentedHighFidelityModel``."""

    def __init__(
        self,
        *,
        area: float,
        slope: float,
        time_step: float,
        initial_level: float,
        initial_flow: float,
        manning_n: float,
        model_cls: type[Any],
        config_cls: type[Any],
        boundary_type_cls: type[Any],
    ) -> None:
        if float(time_step) <= 0.0:
            raise RuntimeError("segmented_hf backend requires positive time_step")
        bottom_width, side_slope = _select_trapezoid_shape(
            depth=max(float(initial_level), 0.5),
            target_flow=max(float(initial_flow), 0.0),
            slope=max(float(slope), 1e-5),
            manning_n=max(float(manning_n), 1e-4),
        )
        depth = max(float(initial_level), 0.5)
        top_width = bottom_width + 2.0 * side_slope * depth
        length = max(500.0, float(area) / max(top_width, 1.0))
        config = self._build_config(
            length=length,
            depth=depth,
            initial_flow=float(initial_flow),
            bottom_width=bottom_width,
            side_slope=side_slope,
            external_dt=float(time_step),
            config_cls=config_cls,
        )
        self.external_dt = float(time_step)
        self.sub_steps = max(1, int(np.ceil(self.external_dt / config.dt)))
        if self.sub_steps > 500:
            raise RuntimeError("segmented_hf backend requires too many sub-steps for this scenario")
        self.model = model_cls(segment_id="STRICT_SEG_000", length=length, config=config)
        self.boundary_type_cls = boundary_type_cls
        self.last_outflow = float(initial_flow)
        for idx, section in enumerate(self.model.sections):
            section.bottom_width = bottom_width
            section.side_slope = side_slope
            section.manning_n = float(manning_n)
            section.bed_elevation = -max(float(slope), 1e-5) * idx * self.model.config.dx
        self.model.reset(initial_level=float(initial_level), initial_flow=float(initial_flow))
        initial_state = self.model.get_state()
        self.downstream_level = float(2.0 * float(initial_level) - float(initial_state.mean_level))
        self.name = "segmented_hf"

    @staticmethod
    def _build_config(
        *,
        length: float,
        depth: float,
        initial_flow: float,
        bottom_width: float,
        side_slope: float,
        external_dt: float,
        config_cls: type[Any],
    ) -> Any:
        min_nodes = 5
        max_nodes = 21
        cfl_target = 0.8
        preferred_dt = min(15.0, external_dt)
        wetted_area = _trapezoid_area(bottom_width, depth, side_slope)
        velocity = abs(float(initial_flow)) / max(wetted_area, 1e-6)
        wave_speed = np.sqrt(9.81 * max(depth, 0.1))
        characteristic_speed = max(1.0, 1.25 * (velocity + wave_speed))
        required_dx = characteristic_speed * preferred_dt / cfl_target
        if required_dx <= 0.0:
            required_dx = length / (max_nodes - 1)
        max_supported_nodes = int(np.floor(length / required_dx)) + 1
        num_nodes = max(min_nodes, min(max_nodes, max_supported_nodes))
        dx = length / max(num_nodes - 1, 1)
        dt_internal = min(preferred_dt, cfl_target * dx / characteristic_speed)
        dt_internal = max(1.0, dt_internal)
        return config_cls(num_nodes=num_nodes, dt=dt_internal)

    def step(self, q_in_command: float, q_out: float) -> float:
        upstream = {"type": self.boundary_type_cls.DIRICHLET_FLOW, "value": float(q_in_command)}
        current_state = self.model.get_state()
        # A fixed downstream level boundary is much more stable than imposing
        # both-end flow boundaries on this explicit Saint-Venant implementation.
        self.downstream_level = float(0.8 * self.downstream_level + 0.2 * current_state.downstream_level)
        downstream = {"type": self.boundary_type_cls.DIRICHLET_LEVEL, "value": self.downstream_level}
        self.model.set_boundary_conditions(upstream=upstream, downstream=downstream)
        for _ in range(self.sub_steps):
            self.model.step()
        self.downstream_level = float(self.model.h[-1])
        self.last_outflow = float(self.model.Q[-1])
        return self.get_level()

    def get_level(self) -> float:
        state = self.model.get_state()
        return float(state.mean_level)

    def get_volume(self) -> float:
        dx = self.model.config.dx
        return float(np.sum(self.model.A) * dx)

    def get_outflow(self) -> float:
        return float(self.last_outflow)


def build_single_pool_backend(
    *,
    backend: str,
    area: float,
    slope: float,
    time_step: float,
    initial_level: float,
    initial_flow: float,
    manning_n: float,
    tank_cls: type[Any] | None,
    fidelity_cls: type[Any] | None,
    geometry_cls: type[Any] | None,
    segmented_model_cls: type[Any] | None = None,
    segmented_config_cls: type[Any] | None = None,
    segmented_boundary_type_cls: type[Any] | None = None,
    delay_steps: int = 1,
) -> Any:
    """Build a strict-testing hydraulic backend.

    ``backend`` can be ``"auto"``, ``"segmented_hf"``, ``"single_channel"``,
    ``"fidelity"``, or ``"tank"``.
    """

    if backend not in {"auto", "segmented_hf", "single_channel", "fidelity", "tank"}:
        raise ValueError(f"unsupported hydraulic backend: {backend}")

    if backend == "segmented_hf":
        if (
            segmented_model_cls is None
            or segmented_config_cls is None
            or segmented_boundary_type_cls is None
        ):
            raise RuntimeError("segmented_hf backend unavailable")
        return SegmentedHighFidelityBackend(
            area=area,
            slope=slope,
            time_step=time_step,
            initial_level=initial_level,
            initial_flow=initial_flow,
            manning_n=manning_n,
            model_cls=segmented_model_cls,
            config_cls=segmented_config_cls,
            boundary_type_cls=segmented_boundary_type_cls,
        )

    if backend in {"single_channel", "fidelity"}:
        if fidelity_cls is None or geometry_cls is None:
            raise RuntimeError("single_channel backend unavailable")
        return SingleChannelFidelityBackend(
            area=area,
            slope=slope,
            time_step=time_step,
            initial_level=initial_level,
            initial_flow=initial_flow,
            manning_n=manning_n,
            geometry_cls=geometry_cls,
            fidelity_cls=fidelity_cls,
        )

    if backend == "auto":
        if fidelity_cls is not None and geometry_cls is not None:
            return SingleChannelFidelityBackend(
                area=area,
                slope=slope,
                time_step=time_step,
                initial_level=initial_level,
                initial_flow=initial_flow,
                manning_n=manning_n,
                geometry_cls=geometry_cls,
                fidelity_cls=fidelity_cls,
            )
        if (
            segmented_model_cls is not None
            and segmented_config_cls is not None
            and segmented_boundary_type_cls is not None
        ):
            return SegmentedHighFidelityBackend(
                area=area,
                slope=slope,
                time_step=time_step,
                initial_level=initial_level,
                initial_flow=initial_flow,
                manning_n=manning_n,
                model_cls=segmented_model_cls,
                config_cls=segmented_config_cls,
                boundary_type_cls=segmented_boundary_type_cls,
            )
        if tank_cls is not None:
            return SimpleTankBackend(
                simulator=tank_cls(
                    area=area,
                    dt=time_step,
                    delay_steps=delay_steps,
                    initial_level=initial_level,
                    initial_flow=initial_flow,
                )
            )
        raise RuntimeError("no hydraulic backend available")

    if backend == "tank":
        if tank_cls is None:
            raise RuntimeError("tank backend unavailable")
        return SimpleTankBackend(
            simulator=tank_cls(
                area=area,
                dt=time_step,
                delay_steps=delay_steps,
                initial_level=initial_level,
                initial_flow=initial_flow,
            )
        )

    raise RuntimeError("no hydraulic backend available")
