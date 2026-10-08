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

## C — strongest counter-hypothesis

A recurrent address is an ordinary persistent register; external storage is already only a few bytes, and writing/maintaining `m` may add noise and drift without reducing total state.

## U — unresolved

Natural Mamba states, learned selective scans, long contexts beyond 512, online task switching, and accelerator memory bandwidth remain untested.
