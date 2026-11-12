# Parking policy experiment report template

Use the [published evaluation](evaluation/README.md) as the current result source. Fill this template from a saved experiment; retain unfavorable and inconclusive outcomes.

## Decision and scope

State the policy decision, modeled location, baseline, selected metrics, and acceptable affordability and availability trade-offs. Label all results as simulation outputs unless supported by field observations.

## Inputs and assumptions

Record the commit, dependency versions, source hashes, dataset hashes and provenance, CRS, capacity/rate defaults, driver mix, duration distribution, destinations, arrival process, traffic source, and price constraints. Document missing field calibration and road-network connectivity.

## Experimental design

Declare seeds, policy variants, duration, warm-up, time step, and the evaluation criteria before running. Use identical scenario fingerprints within a seed. Keep fresh validation seeds separate from seeds used during development. Fix the algorithm before validation.

Revenue uses accepted entry rates for occupied minutes. Separate the measurement window from warm-up. State the cohort for driver outcomes; retain unresolved arrivals. Define conditional cost, walk, drive, and search means and their sample sizes.

## Results

Link the seed CSV, aggregate JSON, charts, raw-result fingerprints, and reproduction commands. Report baseline and policy means plus paired 95% confidence intervals. Include low-demand losses and cost increases; avoid selecting only favorable seeds or metrics. State whether the interval excludes zero and distinguish seed variability from model uncertainty.

## Interpretation

Explain the supported policy conclusion and the trade-offs. A simulation improvement is not measured city revenue uplift. Synthetic forecast holdouts do not establish real demand accuracy. A small graph does not establish scalability, traffic reduction, or concurrent-user capacity.

## Validation and next decision

Record the actual tests and checks run, the held-out result, remaining uncertainty, and what field evidence would change the decision. Credit Jeremy Cleland, Saif Khan, and Asem Zahran for the CIS 505 team project.
