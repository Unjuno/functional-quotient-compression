# MA-440 — continual SSM skill codes

Status: SCREENING  
Evidence lane: CONTINUAL / STORAGE / RETENTION  
Base commit: `1555074`

## H — falsifiable hypothesis

During sequential acquisition of four related stable dynamical skills, a shared SSM plus small task-specific Mirror coordinates will retain earlier skills with fewer total inference bytes than independent transitions, while matching a shared/private residual control.

> **Mirror insertion:** this experiment adds a per-skill coordinate `m` to one shared stable SSM transition so that continual dynamics can be added without allocating a full transition matrix per skill.

## PA30/PA73 delta

PA30/SETA separates shared and task-unique sparse subspaces; PA73 makes SSM dynamics selective. This screen asks whether low-description View codes can provide a useful shared/private boundary in continual dynamics. Compare a frozen shared base with Mirror codes against shared-only, direct low-rank private residuals, and independent transitions. No sparse-subspace discovery claim is made.

## Protocol

Four stable 4D transition skills are trained in a fixed sequence. After each skill, evaluate all seen skills without revisiting their data. Development worlds 44000/44001 choose LR {0.003,0.01}; fresh worlds 44010/44011/44012, seeds 0/1/2. All conditions receive same updates per skill. Measure per-skill trajectory NRMSE after every stage, forgetting, actual serialized bytes per retained skill, active MAC proxy, and replay wall time.

PASS: final-stage Mirror mean NRMSE <=1.10x independent, average forgetting <=1.10x independent, and <=60% independent bytes per skill; stable transitions. FAIL if a gate misses or direct residual/mask control matches with lower bytes.

## C — strongest counter-hypothesis

Task-specific Mirror coordinates may not protect earlier functions when the shared transition changes; sparse private residuals or masks may be more reliable and cheaper.

## U — unresolved

Natural continual LM tasks, sparse subspace discovery, long streams, and accelerator runtime remain untested.

## Results and decision

**D — FAIL.** Final fresh-stage mean NRMSE: Mirror 0.000092 vs independent 0.000058; shared 0.000093; rank-one residual 0.000094. Mean stage forgetting: Mirror 0.256 vs independent 0.052 (shared 0.275). Mirror payload 2,125B vs independent 2,485B (85.5%), failing the <=60% gate. Stability passed (all spectral radii <0.781).

**FACT:** Mirror only slightly improves forgetting over shared-only and is no better in final error; independent transitions have substantially lower forgetting and final error. Mirror uses 14.5% fewer bytes than independent, not the registered 40% reduction.

**INTERPRETATION:** shared transition adaptation interferes with earlier tasks, and per-skill angles do not isolate the old dynamics enough. The small byte saving does not offset the retention deficit.

**H:** tested whether continual per-skill Views can delay private transition allocation while retaining prior dynamics.

**T:** four sequential related 4D stable skills, 250 updates per skill × batch 64, no old-task replay; LR 0.01 Mirror / 0.003 other methods selected on development; fresh worlds 44010/44011/44012, seeds 0/1/2; final per-skill quality, forgetting, actual payload bytes and stability.

**C:** the task family is a synthetic Givens orbit. A mask-based SETA/Piggyback/PackNet control may preserve skills better and must be tested before any continual-learning conclusion.

**U:** sparse subspace discovery/masks, natural task streams, token-selective Mamba and real continual LM retention remain untested.
