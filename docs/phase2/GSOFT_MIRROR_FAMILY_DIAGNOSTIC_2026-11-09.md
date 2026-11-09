# GSOFT Mirror family diagnostic (2026-11-09)

## Fact

- MA-704 tested synthetic 64-wide task transforms using 1/2/4 group-shuffle stages over one shared matrix. Two stages matched dense output within 2.7e-8 at 37.8% independent bytes and 1.25x MAC proxy. Native GSOFT was function/payload-identical. Local NumPy matrix-free wall time was about 28x dense apply.
- MA-705 tested 16 routed logical experts from one shared 64-wide FFN. One stage used 12.7% of independent expert bytes and 1.0625x MAC proxy with exact generated-target outputs. Same-budget IA3/rank-one controls missed quality; native GSOFT was exact. NumPy wall time was slower than dense.
- Both target families were generated from GSOFT factors. Neither is trained-task or language evidence.

## Interpretation

Group-shuffle factors offer a useful synthetic storage frontier but are the existing GSOFT parameterization. The measured NumPy path does not translate lower MAC count into lower wall time. No Mirror-specific improvement is established.

## Strongest counter-hypothesis

These favorable outputs are construction-aligned: the target operator bank lies exactly in the tested GSOFT family. A fused accelerator kernel could improve runtime, but that is unmeasured here.

## Queue ruling

Pause MA-706, whose only proposal uses the same GSOFT group-shuffle view coordinates on Q/K/V/head roles, until it specifies a cache-safe or functional mechanism that beats native GSOFT. Keep its status UNTESTED. Continue MA-707, scaled-Cayley recurrent dynamics, as a different structured family.
