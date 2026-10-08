# Model files

`tabicl_soil.ckpt` holds TabICL v2 (28.5 million parameters), trained further for 3,000 steps on synthetic soil-cement tables. Of these tables, 30 % are drawn from the hierarchical strength law and 70 % from a generic generator; Appendix A of the paper gives the procedure. The file is in the format of the released TabICL checkpoint, so it loads with

```python
from tabicl import TabICLRegressor
model = TabICLRegressor(model_path="models/tabicl_soil.ckpt", n_estimators=2)
```

The file is 114 MB and is stored with Git LFS. After cloning, run `git lfs pull`.

The released TabICL v2 and TabPFN 3 weights are not stored here. They are downloaded by their own packages on first use.

The hierarchical law needs no model file. Its constants are in `src/soilcement/data/law_constants.json`.
