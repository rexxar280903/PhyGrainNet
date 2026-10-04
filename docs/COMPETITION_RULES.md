# Competition Rules Audit

Verified on the live Kaggle pages on 2026-10-04. Re-check the Overview and Discussion pages weekly.

| Item | Verified value |
|---|---|
| Title | Predicting Soil Grain Size Distributions from Images |
| Host / sponsor | Lukas Leibold, Enrico Soranzo — BOKU Vienna, Institute of Geotechnical Engineering (EU project GRID) |
| Type | Community prediction competition (CSV upload; not a code competition) |
| Start / close | 2026-05-18 / **2026-11-30 18:00 WIB (11:00 UTC)** |
| Prize | $500 (300 / 150 / 50); no Kaggle points or medals |
| Team size | max 5; mergers allowed |
| Submissions | 5 per day; select up to 2 finals |
| External data and models | Allowed if publicly and freely available to everyone (2.6) |
| AutoML | Allowed with a proper licence |
| Winner obligations | Deliver training + inference code, environment description, method write-up; non-exclusive licence to the sponsor |
| Hand labelling | Prohibited for validation/test data (Foundational 4.b) |
| Data licence | CC BY 4.0; competition data for non-commercial/academic use |

## Metric

`EMD = Σ_{i=1..10} |F_i − F̂_i| · (log10 x_{i+1} − log10 x_i)` over the 11 points 0.002 … 200 mm. Mean over samples, range [0, 500], lower is better. Implemented exactly in `phygrainnet.metrics.kaggle_emd.emd_per_sample`.

## Submission validity (rejected otherwise)

- header `sample_id,0.002,0.0063,0.02,0.063,0.2,0.63,2,6.3,20,63,200`
- one row per test id, none missing, no extras
- numeric, within [0, 100], non-decreasing
- **exactly 100** in the 200 mm column

`scripts/validate.py` checks all of these.

## Leaderboard

- Public LB = 3 fixed test soils (host-confirmed); private = the other 7.
- Overview says `Final = 0.30 · public + 0.70 · private`; Kaggle's Foundational Rules say winners are decided by the private LB only, and Foundational Rules override competition-specific ones. A clarification thread was opened on 2026-10-02 — watch it.
- The public LB has been reconstructed by probing (top score 0.00001; the author reports ≈10 for his real model). Honest image models are in the ~10–30 range.

## Our stricter rules

See `docs/DECISIONS.md`: DEC-001 (no external weights), DEC-006 (no probing, no searching for test-site lab results).
