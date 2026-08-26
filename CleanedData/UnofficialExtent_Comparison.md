# Official vs. Unofficial Extent — Comparison

Observation/species counts are 5-min-deduplicated within each polygon, so the
only difference between columns is the boundary. Scripps = upland only (MPA excluded).
`imputed_missing` = species the model predicts present in the reserve but with zero
observations there (the 'missing species' output), under the unofficial extent.

| reserve | donor | official_obs | unofficial_obs | obs_delta | official_species | unofficial_species | species_delta | candidate_pool | imputed_missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scripps | TorreyPines | 3143 | 3158 | 15 | 765 | 771 | 6 | 4004 | 331 |
| ElliottChaparral | MissionTrails | 1799 | 2047 | 248 | 571 | 621 | 50 | 6515 | 699 |
| LosMonos | BuenaVista | 1520 | 1532 | 12 | 557 | 561 | 4 | 952 | 10 |
| MissionBay | TijuanaRiver | 1351 | 3116 | 1765 | 396 | 701 | 305 | 2880 | 328 |
