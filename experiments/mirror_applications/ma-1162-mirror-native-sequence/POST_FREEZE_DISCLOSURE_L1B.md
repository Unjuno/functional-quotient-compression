# MA-1162 L1b protocol chronology / correction

The 800-update *number* was motivated by the development-world (618200) retrieval pilot, which ran before the L1v1 short-budget fresh evaluation. However, the decision to launch a separate L1b experiment was taken after inspecting L1v1 fresh results. Consequently, the sentence in `PROTOCOL_L1B_FROZEN.json` saying the 800-update experiment was chosen "prior to seeing the first fresh 96-update outcome" must **not** be read as a claim that the L1b experiment itself was preregistered before L1v1 fresh inspection.

The L1b code, 800-update count, 3 task definitions, comparator list, success threshold, and fresh-world IDs 618211–618213 were fixed **before opening any L1b worlds**. No L1b-world result was used to change them. L1v1 and L1b are two distinct studies, not a pooled single preregistration. The L1b source hash and protocol hash are retained, not retrospectively edited.

This disclosure corrects historical interpretation only; it changes no dataset, learning parameter, output, or gate.