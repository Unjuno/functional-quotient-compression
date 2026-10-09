# State-space role/view family diagnostic — 2026-10-09

## Decision

Pause MA-437 and later state-space role/view candidates that use the same per-role coordinate on shared transition dynamics. Resume only after the insertion is redesigned to escape native-parameter aliasing and to beat the registered compute/storage gates.

## Facts

- **MA-434 training screen:** four-role selective recurrence; Mirror improved fresh sequence NRMSE versus independent (0.02235 vs 0.03894), but actual payload was larger (2,837 B vs 2,649 B) and unfused training time was 9.026 s vs 0.873 s. The registered storage and runtime gates failed.
- **MA-434 direct-parameter diagnostic:** on a minimal selective diagonal recurrence, a scalar role coordinate produced distinct role outputs, but matched a native additive log-decay bias exactly in function and serialized payload (2,457 B). Full-copy compression was only 0.930x, below the 0.80x gate. Fresh stayed sealed after the development gate failed.
- **MA-436 token-wise logical experts:** an earlier synthetic run showed a quality/storage point (2,529 B vs 3,045 B independent) but used 192 vs 128 MAC/token and 1.56x inference time.
- **MA-436 native-alias diagnostic:** an amended development screen found the viewed transition equivalent to a native generated-A parameterization. Payload was 0.743/0.738x full copies, exact native output/payload alias was present, and throughput/diversity gates failed. Fresh stayed sealed.

## Interpretation

The positive synthetic storage/quality observations are bounded feasibility signals. The two direct-control diagnostics show that the current role coordinates can be ordinary native SSM parameters, while the computed view adds recurrent cost and does not meet strict storage/runtime gates. This is sufficient to pause this shared-transition role family under the experiment stop rule; it does not establish that every SSM view is ineffective.

## Hypothesis for any restart

A redesigned SSM view must beat a byte- and function-matched native transition parameterization, retain useful role diversity, and pass actual serialized storage plus recurrent compute/latency gates before fresh worlds are opened.

## Queue

Current worker queue resumes at **MA-443**, outside the paused shared-transition role family.
