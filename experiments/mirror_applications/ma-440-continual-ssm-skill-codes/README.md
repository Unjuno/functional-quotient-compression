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
