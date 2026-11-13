# MA-990 — RANF HRTF personalization feasibility

Status: UNTESTED (deferred; no experiment was run)
Branch: `research/ma-990-ranf-hrtf-personalization-20261113`

## H / T / D / C / U

**H:** A compact listener-specific coordinate on a shared RANF neural field can improve measured-direction HRTF quality per serialized byte over RANF retrieval and native sparse adaptation.

**T (feasibility only):** Inspected the official MERL `ranf-hrtf` source at commit `774561a3683271e1b83cccd15fb5f92086002d55`, its run scripts, and the repository's PA292 notes. The source expects the SONICOM HRTF dataset, which is not present in this workspace; no dataset file or authorized downloadable artifact was identified. Required Python packages `sofar` and `pysofaconventions` are absent. No training, evaluation, or benchmark measurements were run.

**D:** UNTESTED / deferred, not FAIL. The registered experiment cannot be evaluated without its subject-level HRTF data.

**C:** A public SONICOM distribution path may exist outside the sources checked; if supplied or located, the registered RANF comparison may be resumed.

**U:** All model quality, payload-byte, adaptation, retrieval, and runtime claims remain unmeasured.

## Decision

- **FACT:** Official source code was found; required SONICOM measurements and data dependencies were not available locally.
- **INTERPRETATION:** A synthetic HRTF screen would not establish the registered held-out-listener claim and is not substituted here.
- **HYPOTHESIS:** The experiment remains open for a later dataset-backed run.
