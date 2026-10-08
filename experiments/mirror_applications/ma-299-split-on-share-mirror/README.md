# MA-299 — Split-on-Share Mirror code allocation

Status: **FAIL for Mirror-specific advantage; shared allocation mechanism demonstrated**  
Branch: `research/ma-299-split-on-share-mirror-20261008`  
Base commit: `28194ea`  
Prior art: PA30, PA16

## H — hypothesis

Trial a compact Mirror coordinate before allocating a full task-private map. It should keep aligned incoming tasks shared, split on unrelated functions, and improve the byte/quality/compute frontier over native no-view, two-coefficient, and rank-1 residual split controls.

## T — execution

A deterministic NumPy CPU stream has six 16×16 linear tasks: one shared root; three Givens-aligned tasks; one unrelated matrix; one near-aligned map with a small private residual. Development worlds 29901/29902 chose and froze normalized-MSE threshold 0.01; fresh worlds 29921–29923 used 256 train, 128 validation and 512 held-out examples/task. Mirror angle fit used a fixed 4097 point grid; other fits used least squares or rank-1 SVD. All decisions used training and validation only. There were zero optimizer updates.

Controls: hard tie, no-view split-on-share, two-coefficient shared-basis split-on-share, rank-1 residual split, Mirror split, always-Mirror, and independent full maps. Typed binary payloads include complete method/task metadata, shared/private maps, codes, factors, field names, shapes and headers. All final quality and timing replay from deserialized inference payloads. Each policy freezes its shared map, so retention change is zero by construction; this is not an online-learning interference test.

## D — decision

The Mirror policy accepted four view codes (three aligned tasks plus the near-aligned task), split one unrelated task into private weights, and produced a mean 2,337 byte payload. Mean normalized held-out MSE was 0.0005970, maximum over tasks/worlds was 0.0040706. Independent-full used 6,417 B and had effectively zero MSE; native no-view split-on-share used 6,414 B because it privately stored all five later tasks. Thus Mirror delayed physical growth for the aligned task family and used 0.364× independent-full bytes.

The two-coefficient control made the same four sharing decisions and one private split, with mean 0.0005962 MSE at 2,352 B. The Mirror payload saved only 15 B total (0.64%) across the six-task stream, below the preregistered 10% Mirror-specific gate. Its fit compute proxy averaged 1,441,952 vs 827,392 for the coefficient control, and measured fit wall time averaged about 1.30 ms vs 0.99 ms. Mean inference throughput was 34.8M vs 31.1M examples/s on this CPU screen; this timing is noisy and does not offset the very small byte difference or higher fit work. Rank-1 residual fell back to private matrices; always-Mirror without fallback had mean normalized error 0.300.

Therefore the generic share-before-private idea works on the matched stream, but the Mirror coordinate did not materially improve over a two-coefficient address. **FAIL** is the status for the registered Mirror-specific claim.

## Fact / interpretation / hypothesis

**Fact:** Three fresh streams produced the allocation and quality figures above. All 21 method/world payload hashes and MSE values replay exactly; five tests pass. One unrelated task was private in each Mirror stream. The first fresh attempt (29911–29913) retains valid quality/byte evidence, but its angle-fit compute proxy was invalidated; corrected final compute uses amended fresh seeds 29921–29923.

**Interpretation:** A share-then-private allocator can defer full matrix growth when task functions belong to a compact shared family. Here, a standard two-coefficient basis does the same. One scalar angle saves only 15 B total and costs more fit compute; this does not justify Mirror-specific value.

**Hypothesis:** More complex task-view families may yield a larger code-versus-basis storage gap, but SETA-like sparse shared/private discovery itself is not shown to need Mirror codes.

## C — strongest counter-hypothesis

The two-coefficient control is an exact reparameterization of this Givens family: `W R(m) = cos(m)W + sin(m)WG`. It matches quality and allocation exactly, with a small payload premium and lower fit compute. This is the expected strong counter-control for this teacher.

## U — unresolved

No real continual learner, task router, SETA discovery/routing reproduction, online weight updates, natural task stream, GPU kernel, language model, or capacity frontier was tested. Near-aligned tolerance was selected on development worlds only.

## Amendment provenance

A post-fresh compute audit found the angle-fit proxy omitted four of five post-root task fits. Fresh quality/byte data from 29911–29913 are retained in `source/fresh_quality_bytes_valid_compute_invalid.json`, with only their compute values excluded. The corrected proxy and seeds 29921–29923 were frozen before the amended rerun, retained as `source/fresh_final.json`.
