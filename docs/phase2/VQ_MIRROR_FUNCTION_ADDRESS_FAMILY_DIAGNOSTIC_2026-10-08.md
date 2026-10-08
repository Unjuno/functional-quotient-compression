# VQ Mirror function-address family diagnostic — 2026-10-08

## Scope

MA-481 and MA-482 tested discrete functional addresses on the same synthetic rank-four bank of 192 linear maps. MA-481 used one-stage VQ; MA-482 used residual VQ. Both had two frozen development seeds, an exact independent matrix upper, continuous and int8 controls, full uncompressed payload accounting, heldout output replay, and fresh seeds sealed.

## Facts

- **MA-481:** VQ16/64/128 used 5,771/6,539/7,564 B and heldout RMSE ranges 0.1044–0.1179 / 0.0744–0.0859 / 0.0668–0.0747. Distinct heldout address fractions were 0.219–0.234 / 0.594–0.672 / 0.719–0.750. Int8 used 6,101 B with RMSE <=0.00157; continuous codes used 8,149 B with effectively zero error. Every native VQ bank was byte/hash/output identical to Mirror VQ.
- **MA-482:** RVQ8x4 used 11,934 B, address uniqueness 0.969 and RMSE 0.04400/0.05002. RVQ16x4 used 12,450 B, uniqueness 1.0 and RMSE 0.03118/0.04038. Int8 used 6,030 B with RMSE 0.00281/0.00129. Single VQ128 used 8,832 B and RMSE 0.06852/0.08420. Each native RVQ serialization and output was exactly identical to its Mirror-labelled counterpart.
- Both experiments replayed from serialized payloads with zero metric difference; both independent full-matrix uppers reproduced the target functions. No fresh seed was accessed.

## Interpretation

Residual refinement improves the quality-rate curve relative to one-stage VQ on this synthetic coefficient family, with increasing center/index bytes and fit operations. It does not beat int8 on the preregistered combined storage/quality gate. The exact native parameterization aliases explain both experiments: a sequence of learned discrete codebook indices is ordinary VQ/RVQ state, and adding a Mirror label changes neither function nor bytes.

## Family ruling

Pause unchanged codebook-only logical-function candidates MA-483, MA-484 and MA-485 pending a redesigned hypothesis. Resume only with a materially different task and insertion claim that tests behavior beyond native VQ/RVQ—for example routed expert utility versus ordinary MoE at matched actual bytes, with native codebook/expert controls. Do not infer expert capacity from nominal code sequence counts. Other function-sharing families such as sparse dictionary/LISTA (MA-486/487) remain eligible because they test a different representation family.

## H / T / D / C / U

- **H:** Discrete one-stage and residual codebook indices create a Mirror-specific functional family that improves usable logical-function storage over native code controls.
- **T:** MA-481 and MA-482; rank-four synthetic matrices, 128 train/64 heldout, two development seeds each; continuous, int8, independent-matrix and native VQ/RVQ controls; actual NPZ bytes and heldout output replay.
- **D:** FAIL for Mirror-specific attribution in both; MA-482 shows a standard residual-VQ quality-rate improvement over one-stage VQ but fails the int8 quality/storage gate. Fresh data remained sealed.
- **C:** The low-rank smooth bank favors continuous/int8 coefficients, while a different application such as routed experts may expose useful discrete specialization that a synthetic linear bank cannot measure.
- **U:** Natural or trained model tasks, routed expert utility and larger codebook/rate frontiers are not established. Any resumed experiment must freeze a new, non-aliased hypothesis before development runs.
