# Do not duplicate established adjacent experiments

This research combines, but does not rename or replace, earlier experiments:

- **MA-231 / SM002 folding**: fixed Mirror address can be folded into ordinary expert matrices. It trades higher resident RAM for lower execution latency, and often wins eager CPU timing. [SFM002 control](../REPORT.md) separately validates this exact trade.
- **MA-253**: final-FFN placement may leave prefix KV unchanged; earlier FFN changes can invalidate downstream cached KV. A final output View is not necessarily a full global Mirror.
- **MA-691**: known transform K_m=K A_m, V_m=V B_m allows a canonical cache with query/readout transforms. Cache **storage** reuse alone does not eliminate attention score compute for all roles.
- **MA-310 / MA-351, PA42**: MIMO and single-forward multi-output models exist; a Mirror single-forward family must outperform their ordinary shared-head/gate controls.
- **MA-255 / PA16**: Parameter Superposition is pre-existing many-logical-model sharing.
- **PA434–436**: predicting/offloading the next expert is an established systems problem (Pre-gated MoE, SpecPrefetch, SPICE).

**Full-fresh data firewall and word choice:** A source-aligned synthetic PASS establishes only *mechanics*. A native folded or simple linear control that equals Mirror is M0, not an invented capacity multiplier. Real measured CUDA transfer overlap is mandatory for MA-1176 adoption.

**Next tests:** (1) naturally trained task bank and causal routing, (2) cheap native output-head factorization matched byte, (3) hot-fold cache threshold based on actual RAM, (4) GPU two-stream timing after confirmed expert predictions.
