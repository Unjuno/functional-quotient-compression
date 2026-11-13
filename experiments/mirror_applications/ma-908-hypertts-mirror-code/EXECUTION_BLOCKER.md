# MA-908 execution boundary

## H

Constrain HyperTTS's generated adapter state to an explicit small Mirror code and test whether it matches or beats the native equal-capacity hypernetwork on heldout speaker quality, generation cost and actual code/adapter bytes.

## T

Reviewed PA254 and cloned official `declare-lab/hypertts` at commit `2e2b1a1d284849a903d41c005bb32dff9202a158`. Downloaded its linked 600k LTS100 checkpoint (SHA-256 in `source/UPSTREAM.md`). Inspected configs, preprocessing manifests, model adapters, requirements and local assets. The repo has no raw VCTK/LTS audio or precomputed speaker embeddings needed for heldout quality; runtime is CPU-only and official dependency pins target a much older stack.

Required controls: native HyperTTS generator, equal-capacity hypernetwork, Mirror-code-constrained generator, and independent adapter bank; compare actual serialized inference payloads, heldout speaker similarity/intelligibility/acoustics, generator compute and RTF. No permitted fresh speaker quality set is available.

## D — UNTESTED

Source and one pretrained checkpoint exist, but needed reference speech and speaker embeddings do not. No training, synthesis, payload measurement, or quality score was run. Status remains UNTESTED, not FAIL.

## C

A code-only bottleneck may merely constrain the existing hypernetwork and lose speaker-specific information; if it matches, the result may be native hypernetwork compression rather than Mirror-specific value.

## U

Heldout speech data, compatible pretrained speaker embedder, and the Mirror-vs-equal-capacity-HyperTTS quality/bytes/compute frontier remain unresolved.
