# MA-475 status

- Status: PROMISING for aligned memory-value storage only
- Branch: `research/ma-475-serac-mirror-values-20261009`
- Base commit: `69c70957`
- Protocol freeze: `510715fb`
- Development complete: yes
- Fresh/audit opened: yes, after protocol freeze
- Results committed: yes (`4ff1c6f0`)
- Verification committed: yes (`4ff1c6f0`)
- Registry row updated: yes

## Next action

MA-475 is complete; proceed with MA-476 on its dedicated branch.

## Limitations

- The value bank is a known fixed-norm 2D orbit; no natural SERAC values or learned counterfactual model were tested.
- Mirror is only 2.6% smaller than the generic Cartesian coefficient control at N=64 and ties it at N=8.
- Lookup wall time starts after bank values are decoded; bank reconstruction cost is not included in this lookup metric.
- The exact nearest-cosine scope classifier is shared by all methods and is not SERAC's learned classifier.

## Decisions / rulings

- Increased timing probe repetitions after an initial run produced an isolated high Mirror lookup time. The data, threshold, and quality/storage metrics stayed frozen; all fresh rows were rerun with the same worlds and seeds.
- The code's main comparison is external value representation; retrieval accuracy is independently measured and held identical.
