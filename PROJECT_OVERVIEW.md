# Parking policy experimentation

A CIS 505 Algorithms Analysis and Design team project at the University of Michigan–Dearborn, Summer 2025, by Jeremy Cleland, Saif Khan, and Asem Zahran.

The current implementation supports repeatable comparisons of bounded occupancy-based pricing and directed-graph routing. It includes a local browser dashboard, destination-aware driver choices, capacity-aware arrival forecasts, seeded simulated traffic, optional Apple MLX batch-distance computation, and inspectable reports.

## What is evaluated

The [dated evaluation](docs/evaluation/README.md) measures simulated revenue, capacity-weighted occupancy, parking success, and quoted stay cost across low, medium, and high offered demand. Static and dynamic policies receive matching scenario inputs for each seed. Five original seeds are followed by ten fresh seeds with the policy frozen. Effects use paired 95% Student t intervals and are reported alongside affordability and availability trade-offs.

The experiment uses eight hours per scenario, excludes two warm-up hours, and bills each stay at its accepted entry rate. Late unresolved arrivals remain in driver-outcome denominators. A two-hour warm-up does not prove equilibrium for long-stay commuters.

## Components

- Pricing: common pre-update snapshot, capacity-aware demand signal, district adjustment, bounds, and movement limits. This is a heuristic; equilibrium or optimality is not established.
- Routing: directed NetworkX shortest paths, distance and travel-time weights, expiring traffic observations, and separate straight-line walking estimates.
- Forecasts: deterministic historical arrival tables normalized by actual zone capacity. Accuracy is evaluated on synthetic history only.
- Simulation: explicit clock, independent random streams, heterogeneous duration and preference assumptions, and a working synthetic offline road network.
- Reporting: saved zone observations, seed-level policy comparisons, CIs, source/input fingerprints, and exportable local dashboard results.
- Compute: NumPy baseline, optional MLX for batch walking distances, CPU graph searches, and independent process workers.

## How to inspect it

Follow [README.md](README.md) for the pinned Python setup and CLI commands. Start the local dashboard with `.venv/bin/parking-optimize --mode dashboard --port 8765`. A synthetic demo needs no API keys or map download. The [public report](https://jeremy-cleland.github.io/parking_optimization/) provides the current charts and evidence downloads.

## Remaining work

Calibrate arrivals, durations, destination and price preferences against field observations; establish dataset provenance; validate pedestrian accessibility and provider traffic; evaluate a larger connected road network; and select explicit revenue, availability, and affordability objectives before a field trial. Strict Mypy checking still reports legacy errors. No municipal deployment, concurrent-user benchmark, or field outcome is claimed.
