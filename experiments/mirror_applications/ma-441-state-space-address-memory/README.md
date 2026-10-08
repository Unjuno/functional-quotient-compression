# MA-441 — state-space address memory for Mirror codes

Status: SCREENING  
Evidence lane: DYNAMIC FUNCTIONAL STATE / RETENTION / STORAGE  
Base commit: `286f75c`

## H — falsifiable hypothesis

A small Mirror address stored in recurrent SSM state can retain a selected logical transition through long distractor sequences and later recover the selected function with lower state bytes than an external per-task routing table, without materially greater drift than an ordinary fast-weight memory.

> **Mirror insertion:** this experiment adds a persistent two-coordinate `m` to the recurrent SSM state so that one physical transition can recall a selected logical function after intervening tokens without an external address table.

## PA25 / PA73 delta

PA25 interprets recurrent systems as fast-weight programmers; PA73 makes SSM propagation selective. This screen isolates whether an address itself can live in SSM state. The direct controls are an external address register and a simple fast-weight/register baseline; no Mamba language-model claim is made.

## Protocol

Synthetic recall task with four stable 4D logical transitions. An initial cue selects one transition, followed by distractor tokens, then a query token requires the selected dynamics. Sweep delay lengths 8, 32, 128, and 512. Compare: external exact address register; recurrent Mirror address with learned write/retention; ordinary recurrent scalar/vector register with equal state width; and a rank-one fast-weight memory control. Development worlds 44100/44101 choose LR {0.003,0.01}; fresh worlds 44110/44111/44112, seeds 0/1/2. Report recall NRMSE, drift vs delay, exact inference state bytes (including stored m/register/fast weight), MAC proxy, and sequence wall time.

PASS: at delay 512, Mirror recall NRMSE <=1.10x external register, <=60% its actual state bytes, and no more than 1.25x MAC; drift must not increase by >10% from delay 8. FAIL if these gates miss or ordinary register/fast-weight control matches at lower bytes.

### Amendment A1

An initial dev/fresh run exposed a setup defect: the learned retention parameter started at alpha=0.018, causing immediate address erasure and chance-level recall. A1 changes only the initial retention logit to 6 (alpha=0.9975), preserves all data worlds, seeds, optimizer budgets, and gates, and invalidates the initial fresh rows.

### Amendment A2

A post-run audit found a metric mismatch: the frozen contract names recall NRMSE, while the runner used NLL for primary selection/results. A2 changes primary metric to probability-vector NRMSE against the one-hot target; NLL and accuracy remain secondary diagnostics. Development and fresh are rerun on the same worlds/seeds with no gate changes.

## C — strongest counter-hypothesis

A recurrent address is an ordinary persistent register; external storage is already only a few bytes, and writing/maintaining `m` may add noise and drift without reducing total state.

## U — unresolved

Natural Mamba states, learned selective scans, long contexts beyond 512, online task switching, and accelerator memory bandwidth remain untested.

## Results and decision

**D — FAIL.** Corrected fresh Mirror accuracy remained 100% through delay 512, but probability NRMSE rose from 0.00419 at delay 8 to 0.01559 at delay 512. The external exact address had zero error at every delay. Actual payload was 2,920B for Mirror vs 1,938B external; recurrent address state was 8B vs 4B. Equal-width 4D register also retained 100% accuracy and had lower NRMSE than Mirror, while rank-one fast-weight memory degraded to 94.4% accuracy at delay 512.

**FACT:** Mirror retained the correct role decision for every tested example at every delay, but confidence drift increased. It used more state bytes and more serialized bytes than the external route control. The ordinary register was a stronger recurrent-memory control.

**INTERPRETATION:** recurrent `m` can serve as persistent address memory in this synthetic task, but it does not compress the exact external address or outperform a generic same-purpose register. This is a negative storage/retention Pareto result despite perfect hard-decision recall.

**H:** tested whether address `m` can live in recurrent state through distractors at lower total state bytes than an external route table.

**T:** 4 logical functions, delays 8/32/128/512, 500 updates × batch64 for learned methods; development worlds 44100/44101 selected LR 0.01 for Mirror/register/fastweight; fresh worlds 44110/44111/44112, three seeds; amended primary probability NRMSE with accuracy secondary. External exact route is a deterministic no-training baseline.

**C:** this is a short synthetic role-recall task and does not test Mamba selective state or learned online routing. A custom two-coordinate code may underperform a discrete ID/register by construction.

**U:** natural recurrent state, online task changes, longer delays, non-oracle cue extraction, and accelerator memory bandwidth.

### Protocol amendments

- **A1:** retention initialized at alpha=0.018 erased cues; changed initial retention to alpha=0.9975, keeping worlds/seeds/gates fixed. Initial fresh rows invalidated.
- **A2:** corrected primary metric from an accidental NLL implementation to the registered probability-vector NRMSE; NLL/accuracy remain diagnostics. Initial A1 fresh rows invalidated and rerun. See both `artifacts/audit_amendment_A*.json` records.
