# T248–T265: selected synthetic findings, not real-model evidence

Source: the supplied T265 handoff. Archive and original report hashes are in
`provenance/t282/SOURCE_ARCHIVES.json` and `T265_REPORT_INDEX.json`.

- T248–T253: equal-byte MLP-family comparisons and support-count selection. Maximum support can overcorrect; K1024-first is not a justified universal policy.
- T254–T259: joint eight-branch objectives, byte reallocation and output-difference gains. Added search complexity was not uniformly better on the synthetic audit worlds.
- T260–T265: conditional output-gain compilation by duplicating only corrected neurons, and separate FC/PROJ gains. The reported 960/16,384 case corresponds to 5.859375% additional theoretical MLP linear MACs, not measured Transformer/GPU overhead. Separate gains improved 7/8 fresh synthetic worlds in the reported comparison, not every world.

Preserve these design lessons and negative cases, but do not present them as trained-Transformer quality or native compressed execution. The original scripts and complete historical outputs remain in the source archive; they are not all promoted into the T282 scalar-baseline lane.

The scientifically relevant next test is whether contextual sharing/private allocation adds value over a strong non-sharing whole-model control at the same actual final bytes. The current curated runner reconstructs the control; it does not implement that ablation automatically.
