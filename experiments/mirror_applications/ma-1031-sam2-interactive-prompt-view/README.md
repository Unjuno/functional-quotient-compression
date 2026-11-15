# MA-1031 — SAM2 interactive correction prompt views

Status: SCREENING. Test whether repeated positive/negative user correction clicks can be stored as a smaller object-local prompt View code while the official SAM2 prompt encoder and streaming decoder remain unchanged.

The Mirror code factors each click pair into a per-object shared separation radius plus a per-frame midpoint and rotation angle. Controls include exact direct clicks, a native coarse-grid coordinate code, zero corrections and exact-mask correction upper bound. The simulation uses only current-frame mask error to generate the human-like correction; it does not inspect future frames or future masks. It compresses interaction history and does not claim to reduce the count of clicks.

Protocol, fresh seeds, storage charging and gates are in `PROTOCOL.json`. Result and provenance files will be added after the development radius is frozen and the three audit worlds are run.
