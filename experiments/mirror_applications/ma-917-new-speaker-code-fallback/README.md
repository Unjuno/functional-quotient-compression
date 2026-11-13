# MA-917 — few-shot new speaker code with private fallback

Status: UNTESTED
Evidence lane: CONTINUAL_SPEAKER_ADAPTATION / QUALITY / STORAGE
Base commit: `research/mirror-application-worker-ready-20261007`

## H — falsifiable hypothesis

For each new speaker, a compact Mirror code adapted from a frozen multispeaker generator will reach acceptable similarity and intelligibility for speakers on the shared manifold, while allocating a private adapter only to outliers improves the quality/bytes frontier and preserves earlier voices.

## T — feasibility review

Inspected PA253–PA254 and official HyperTTS source/checkpoint. The only located official checkpoint is LTS100 single-speaker pretraining. The README's VCTK-finetuned checkpoint section is commented out; VCTK configuration expects a multispeaker generator, GE2E embeddings, and a 900,000-step training schedule. Environment is CPU-only. No consenting/local few-shot evaluation set or compatible speaker embeddings are available.

## D — UNTESTED

No new-speaker adaptation, private fallback, retention, speaker-quality, intelligibility, actual-byte, or compute result was produced.

## C — strongest counter-hypothesis

A single code may be too constrained for accent/prosody outliers; private adapters may be required for most speakers and erase the storage gain.

## U — unresolved

Runnable frozen multispeaker generator, compatible speaker encoder assets, consent-safe support/query voices, and a predeclared threshold for triggering private residuals. See [EXECUTION_BLOCKER.md](EXECUTION_BLOCKER.md).
