# Curated T282 local lane

`code/`: eight byte-identical measured core files; see `provenance/t282/SOURCE_ARCHIVES.json`.
`tests/`: relocated artifact-free checks (97 quantizer/schema + 52 envelope = 149).
The historical 153 included one public-transcription manifest check and three large-artifact checks;
those four are not falsely counted as clone-only tests. Artifact regeneration is checked separately.
`data/`: only the eight previously authored calibration probes needed for the frozen scalar fits.
`locks/`: immutable three-candidate/checkpoint/hash contract.
`records/`: selected historical numerical witnesses; these are not new measurements.

Canonical entry point: `scripts/fqc_rebuild_frozen.py` from the repository root.
The new entry point was checked against all three original artifact SHA256 values and tensor hashes.
It performs historical CPU reconstruction, not official validation and not an FQC-sharing experiment.
Some legacy decoder branches remain present for provenance; old 64x encoder helpers are not all ported
here. Use the original source ZIP for historical MLP-tail experiments rather than assuming this lane
is a complete replacement for every prior runner.
