# MA-562 — Factorized width x depth Mirror address

Status: SCREENING. PA108/PA109 recognize width/depth as existing architecture coordinates. This screen tests an extra additive width-plus-depth function code on a shared synthetic elastic network and holds out joint combinations. The generic direct scalar control intentionally applies the same additive law; exact equality means no Mirror-specific advantage. This is a CPU mechanism screen, not a trained OFA or US-Net result.


## Fresh result

Across three fresh seeds and four heldout width-depth cells, Mirror and direct additive scalar control both achieved mean/max NRMSE 3e-12/1.28e-11, with exactly equal outputs and equal 4464 B total serialized payload. Native subnet mean NRMSE was 0.1572 at 4028 B; independent full maps were 1.73e-16 at 12757 B. Thus the additive factor representation interpolates the deliberately matching teacher, but Mirror adds no incremental function or byte benefit over the direct additive control. **Disposition: FAIL for Mirror-specificity.**
