# Sparse function Mirror coordinate diagnostic — 2026-10-08

## Scope

MA-486 and MA-487 test sparse functional coordinates over a shared orthonormal dictionary on synthetic 16x16 linear maps generated with three active atoms. MA-486 evaluates sparse OMP code storage versus dense codes; MA-487 evaluates learned LISTA inference against direct projection/top-3 and OMP. Both use two frozen development seeds, full dictionary bytes, heldout outputs, and sealed fresh seeds.

## Facts

- **MA-486:** sparse-int8 OMP payload was 35,379 B with RMSE 0.000568/0.000563 versus dense-int8 40,090 B at the same RMSE. It missed the frozen <=0.80x byte gate. Sparse decode proxy was 147,456 versus dense 1,572,864 operations; total OMP+decode+query proxy was 5.91M versus 7.34M. Native OMP exactly matched bytes, hashes and outputs.
- **MA-487:** LISTA1/2/4 produced RMSE 0.00158/0.00108/0.00083 and average active atoms 3.0, with about 1.59/1.60/1.62M inference operations versus OMP3 at 4.87M. But direct projection plus top-3 gave RMSE about 1e-7 at 1.57M operations and 33,409 B, dominating all LISTA depths on the measured quality/operation/payload frontier. LISTA predictor payloads were 33,906–33,930 B.
- Independent matrix controls reproduced the target; both experiments passed full serialized output replay with zero maximum metric difference. Fresh seeds remained sealed.

## Interpretation

Sparse coordinates can substantially reduce decoding arithmetic, and sparse-int8 OMP gives a modest byte reduction versus dense-int8 codes when a large shared float32 dictionary dominates the payload. These benefits are standard sparse coding behavior and exactly alias OMP. On this orthonormal dictionary, direct projection and top-3 selection is an even cheaper, more accurate native control than learned LISTA. Neither result demonstrates additional Mirror-specific function capacity.

## Family ruling

Do not continue unchanged sparse-code-only proposals that restate OMP, top-k projection or fixed orthogonal sparse coordinates as Mirror. MA-488 remains eligible because it tests a different question from the same family: whether shared atoms suffice across heterogeneous functions or private atoms are needed. It must compare shared-only, shared/private and independent controls with actual serialized bytes and heldout function behavior. LISTA routes that match direct projection should not be called a LISTA or Mirror gain.

## H / T / D / C / U

- **H:** Sparse shared coordinates or a learned LISTA inference rule add functional multiplicity beyond ordinary sparse coding and simple projection controls.
- **T:** MA-486 OMP sparse storage and MA-487 LISTA1/2/4; two seeds each; independent matrices, dense codes, OMP, direct top-3, serialized payload and replay.
- **D:** FAIL for Mirror-specific attribution. Sparse OMP cuts decode work but misses byte threshold and exactly aliases native sparse coding. LISTA beats OMP compute but is dominated by direct projection/top-3. Fresh sealed.
- **C:** The orthonormal three-sparse toy makes direct projection unusually strong; learned overcomplete/non-orthogonal dictionaries or noisy function descriptors could alter the result.
- **U:** Natural task transfer, learned dictionary formation, noisy descriptors, GPU latency and private atom allocation remain unknown. MA-488 tests private/shared heterogeneity rather than repeating fixed sparse inference.
