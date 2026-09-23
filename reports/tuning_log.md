# Detection tuning log

## Iterations
| Version | Change | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| v1 | Naive rule (any location change within 60 min), random-city data | 414 | 4 | 50.2% | 99.0% |
| v2 | Realistic data (home city + whole-day trips), same naive rule | 31 | 3 | 93.0% | 99.3% |
| v3 | Speed-based rule (>900 km/h), realistic data | 59 | 6 | 87.4% | 98.6% |
| v4 | v3 + return-leg correlation | 20 | 6 | 95.3% | 98.6% |

**Key findings**
- v1 to v2: 411 of 414 false positives came from impossible travel. Root cause was unrealistic
  synthetic data (users teleporting between cities), not the detector.
- v3: speed-based rule (haversine distance / time) is physically correct but also flags the
  "return leg" - the real user logging in from home after a fraudster's foreign login.
- v4: return legs with no other suspicious signal are linked to the existing incident instead
  of raising a new alert. 39 suppressed, 0 of them fraud.
- Note: v1 and v2 used different generated datasets (7,453 vs 7,327 events); v2 to v4 used the same data.

## Isolation Forest contamination sweep (v4)
| contamination | alerts | precision | recall |
|---|---|---|---|
| 0.01 | 287 | 99.7% | 69.1% |
| 0.02 | 291 | 99.7% | 70.0% |
| 0.04 | 337 | 99.1% | 80.7% |
| 0.06 | 428 | 95.3% | 98.6% |
| 0.08 | 556 | 73.9% | 99.3% |

**Chosen: 0.06.** Recall collapses below it and precision collapses above it.
Caveat: 0.06 is close to the true fraud rate (~5.7%), which a production system would not know;
in practice this would be tuned against analyst feedback on closed alerts.
