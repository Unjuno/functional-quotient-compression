# MA-911 execution blocker

## H — falsifiable hypothesis

A compact Mirror speaker × content factorization in an interventional speech representation can preserve and independently manipulate speaker identity/content on unseen swaps and OOD speakers, improving actual representation bytes or quality over the native projection/subspace and ordinary factorized controls. Any apparent gain fails the claim if swaps leak words/identity or if a simple native subspace split matches it.

## T — attempted protocol / feasibility

Reviewed PA257, official repository, configs, training/evaluation code, Zenodo releases, local data and runtime. Faithful controls require the official interventional training set and pretrained WavLM/SSL features; fresh audits require VoxCeleb1 speaker verification, SpeechCommands keyword spotting, and heldout speaker/content swaps. The README generates interventions with F5-TTS in Apptainer from LibriTTS test-clean source audio; the default set is 32 speakers × 256 target sentences (8,192 utterances), with F5-TTS invoked for target utterances. Current workspace contains no audio or feature files, no torchaudio or Apptainer, and no CUDA. Zenodo releases package source only.

## D — UNTESTED

No model training/evaluation or synthetic substitute was run; no bytes, speaker verification, keyword-spotting, swap leakage, or quality result exists. The registered claim remains untested, not failed. Astra reviewed this blocker and agreed that a tiny analytic/feature substitute would change the hypothesis and still miss OOD speaker verification.

## C — strongest counter-hypothesis

The paper's native interventional projection already provides speaker/content separation; a Mirror coordinate may merely re-label its subspaces or add storage, while uncontrolled swaps could retain lexical or identity leakage.

## U — unresolved

Official datasets/features/checkpoints, CPU feasibility at useful scale, heldout intervention performance, causal swap leakage, actual serialized bytes, and Mirror-vs-native factorization remain unresolved.
