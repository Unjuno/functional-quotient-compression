# MA-714 — Fisher-weighted Mirror code composition

Status: SCREENING  
Evidence lane: MECHANISM/QUALITY/STORAGE  
Base commit: `6b6d3bc3`

## Hypothesis

H: Projecting Fisher-weighted merge objectives into a shared task-delta basis improves on unweighted code averaging and retains full Fisher merge quality with a smaller complete payload.

## Mirror insertion

> **Mirror insertion:** merge task coordinates using projected per-task Fisher precision, then retain a compact shared basis and composed code for each requested model.

PA178 motivates Fisher importance weighting; PA26 supplies Task Arithmetic as the unweighted parameter control. Native projected-Fisher code solving is the mandatory attribution control. Protocol gates and paid state are in [PROTOCOL.json](PROTOCOL.json).

## Results

Pending frozen development.

## H/T/D/C/U

- **H:** Fisher geometry in coordinate space can retain task quality while compressing many merged outputs.
- **T:** D=256, 8 tasks, rank8 basis, 16 synthetic merge requests, 4 source tasks each, private fractions 0/.1/.25.
- **D:** Pending.
- **C:** Full Fisher merging may be better because diagonal importance weighting moves outputs outside the shared basis; native projected solve may be identical.
- **U:** Empirical Fishers, fine-tuned model deltas, natural tasks, and optimized runtime.
