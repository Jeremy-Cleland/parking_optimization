# Parking policy experiment presentation

1. **The decision:** when do occupancy-based prices improve simulated parking access, and at what revenue and affordability cost?
2. **The model:** zones and road extract, assumed capacities/rates, driver preferences and durations, destinations, and seeded traffic. Geographic context is not field calibration.
3. **The comparison:** static/dynamic pricing crossed with distance/time routing; matching seeds; eight hours total with two warm-up hours excluded; entry-rate billing.
4. **The evidence:** show the [dated study](evaluation/README.md), original and fresh validation seeds separately, seed-level CSV, paired confidence intervals, and the cost/success trade-off. Include every demand level.
5. **The demonstration:** start the local dashboard; load saved results; inspect occupancy/revenue/success/cost charts; change a scenario; export its report. Explain that a short smoke test cannot fill a city-sized capacity model.
6. **Compute:** show optional MLX batch-distance calculation and CPU graph/decision work. A kernel benchmark is not an end-to-end performance claim.
7. **Limits and next work:** uncalibrated behavior, assumed capacities/rates, incomplete dataset provenance, coarse search delays, disconnected small graph, and no municipal or production-use result.
8. **Team:** Jeremy Cleland, Saif Khan, and Asem Zahran; CIS 505, University of Michigan–Dearborn, Summer 2025.

Use values directly from the published artifacts. Do not substitute hypothetical improvements or claim real-world congestion, latency, scalability, or revenue gains.
