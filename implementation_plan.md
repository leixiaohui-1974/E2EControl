# Implementation Plan - Topology Visualization

## Goal
Implement a dynamic topology diagram in the web interface to visualize the water system structure and real-time status (water levels, flow rates).

## User Requirements
1.  **Topology Diagram**: Show basic information of the water system.
2.  **Real-time Updates**: Update information on the diagram when scenarios/simulation steps update.
3.  **Web Display**: Must be integrated into the existing web system.

## Proposed Changes

### 1. Frontend (`web/index.html`)
*   Add a new section or card for "System Topology".
*   Add a `<canvas>` element or an SVG container to render the diagram.

### 2. Frontend Logic (`web/js/app.js`)
*   **Topology Renderer**: Create a function to draw the system components:
    *   **Pools**: Rectangles/Trapezoids showing water level (fill height).
    *   **Gates**: Vertical lines/blocks between pools.
    *   **Channels**: Lines connecting them.
*   **Data Binding**: Update the drawing loop to use the latest simulation data (`current-level`, `current-flow`, etc.) fetched from the backend.
*   **Animation**: Use `requestAnimationFrame` for smooth water level transitions if possible, or just update on polling tick.

### 3. Backend (`api.py` / `simulation_manager.py`)
*   The existing `/simulation/<id>/status` and `/simulation/<id>/history` endpoints already return `levels` and `flows`.
*   We might need a new endpoint (or static config) to describe the *static* topology (e.g., "Pool 1 connects to Pool 2 via Gate A").
    *   *Assumption*: For this MVP, we know it's a series of pools. We can hardcode the structure (e.g., 3 pools in series) or infer it from the config.
    *   *Action*: I will check `config.yaml` to see if the topology is defined there.

## Verification Plan
1.  **Visual Check**: Run a simulation and verify the topology diagram appears.
2.  **Dynamic Update**: Confirm water levels in the diagram rise/fall according to the simulation data.
3.  **Scenario Check**: Switch scenarios (e.g., "Flood") and verify the diagram reflects the high water levels.
