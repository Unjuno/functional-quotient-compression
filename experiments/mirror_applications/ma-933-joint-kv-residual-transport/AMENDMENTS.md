# Amendments

## Amendment 01 — implementation pilot (not decision-quality fresh-seed evidence)

An initial run used TinyStories rows 512:768 for fitting and 384:512 for validation. It produced 43,091 B chart / 1,609,494 B independent maps and cache NLL deltas of +3.50 to +3.69 versus target native for chart and +0.858 for independent affine. However, code inspection after the run found chart initialization happened before the per-seed RNG reset; the optimizer seed was applied, but initialization was not reproducibly controlled. Preserve these values as implementation-pilot evidence; do not use them as the preregistered multi-seed decision.

## Amendment 02 — frozen correction

Before opening new validation rows, set Python and Torch seeds before model initialization and batch sampling. Use new disjoint train rows 768:1024, fresh validation rows 512:640, and fresh seeds 93321/93327/93333. Architecture, rank, updates, controls, and quality gate are unchanged. No tuning was selected from amendment 01 metric values.
