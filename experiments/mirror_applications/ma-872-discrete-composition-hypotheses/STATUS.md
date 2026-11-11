# MA-872 status

- Status: FAIL (Mirror-specific gate); discrete native-codebook screen passes its scoped quality/storage gate
- Branch: `research/ma-872-discrete-composition-hypotheses-20261111`
- Base commit: `d2afcc2c`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: yes, after protocol freeze and development validation
- Results committed: pending
- Verification committed: pending

## Next action

Commit and push the checked MA-872 evidence, then integrate it into the cumulative research branch.

## Blockers

None.

## Decisions / rulings

- Discrete selection met the quality gate at 2 support examples; continuous rank-2 PCA first met it at 4, independent coefficient fitting at 16.
- The discrete codebook payload is 569 B vs 809 B independent, but 489 B continuous PCA is smaller.
- Native nearest-codebook search exactly aliases Mirror addresses in output, choice, bytes, and decoder operations; no Mirror-specific gain.
- This tests new instances from known prototypes only; unseen rule discovery is untested.
