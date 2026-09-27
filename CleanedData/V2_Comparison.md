# v2 run -- refreshed data, unofficial extents

Observations are from the Sep 14 2026 iNaturalist API pull, cut to each
reserve's unofficial boundary and 5-min deduplicated. `candidate_pool` is the
partner's species list -- the only species that can be predicted.
`imputed_missing` = predicted present at the reserve with zero observations
there. v1 columns are the Jan-Feb 2026 website-export run for comparison;
Kendall-Frost's v1 partner was Tijuana River, not Sweetwater Marsh.

| reserve | partner | study_obs | study_species | candidate_pool | imputed_missing | v1_candidate_pool | v1_imputed_missing | habitat_blind |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scripps | Torrey Pines State Reserve | 4562 | 956 | 4277 | 352 | 4004 | 331 | False |
| ElliottChaparral | Mission Trails Regional Park | 3118 | 870 | 6819 | 805 | 6515 | 699 | False |
| LosMonos | Buena Vista Park | 2968 | 877 | 1077 | 7 | 952 | 10 | False |
| MissionBay | Sweetwater Marsh | 4959 | 905 | 1218 | 23 | 2880 | 328 | False |
