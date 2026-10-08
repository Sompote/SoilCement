# Data dictionary

`corpus_main.csv` holds the 1,372 mixes used in the paper, with replicates averaged. `corpus_all.csv` holds all 1,634 averaged results, and its column `in_main_set` marks the main set. The results left out of the main set are sand-added blends, peats and organic soils, construction-waste slurry, compacted silt, and vane strengths below 15 kPa.

The `flag` column holds notes from the compilation. Some notes give the reason for exclusion. Others are cautions on results that remain in the main set, such as `laboratory-made soil` for the Virginia Tech clays, or notes on assumed water contents and rounded values.

| column | meaning |
|:--|:--|
| `deposit` | deposit, the unit held out in the paper's protocol |
| `clay_as_reported` | clay name as given by the source |
| `source_id` | source key (see `sources.csv`) |
| `laboratory` | laboratory group: sources with shared authors or institution are merged |
| `country` | country of the deposit |
| `binder`, `binder_class` | binder as reported; class OPC, PCC or slag cement |
| `water_content_pct` | water content at mixing, % of dry clay |
| `binder_pct` | binder content, % of dry clay |
| `pozzolan_pct_of_cement` | pozzolan added, % of cement |
| `equivalent_cement_pct` | λ × binder × (1 + 0.75 × pozzolan/100), with λ = 1 (OPC, PCC) or 1.72 (slag cement) |
| `curing_days` | curing time, days |
| `LL_pct`, `PL_pct`, `PI_pct` | liquid limit, plastic limit and plasticity index of the base clay, % |
| `qu_kPa` | unconfined compressive strength, kPa (laboratory vane strength for some very soft mixes; see `test`) |
| `test` | UCS or vane |
| `recovery` | how the value was recovered: table, text or vector figure |
| `n_replicates` | number of reported values averaged |
| `flag` | compilation note: reason for exclusion, or a caution on an included result (empty if none) |
| `in_main_set` | `corpus_all.csv` only: True for the 1,372 mixes of the main set |

`sources.csv` lists, for each source, its kind, country, deposits, number of mixes, recovery method and test.
