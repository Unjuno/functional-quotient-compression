# MA-296 status

- Status: FAIL
- Branch: `research/ma-296-orthogonal-task-vector-mirror-20261009`
- Protocol freeze commit: `eb49e0cc`
- Fresh/audit: complete, 3 worlds × 3 seeds
- Verification: complete
- Registry / claim ledger: updated in this commit

## H / hypothesis
A compact address can recover correlated task deltas from a shared superposition with lower interference and fewer serialized bytes than independent task vectors.

## T / execution
Synthetic linear tasks, D=512 and N=8; 640 support / 128 query examples per task. Compared unbound sum, independent deltas, random PSP codes, Hadamard address, and generic QR basis. Fresh worlds 29610–29612, seeds 0–2.

## D / decision
FAIL. Hadamard and generic QR match raw deltas at about 2e-6 query NRMSE, but use 16,735 B and 16,744 B versus 16,473 B raw. Unbound sum uses 2,131 B but query NRMSE is about 2.52. Random PSP codes yield about 0.94 query NRMSE with this fixed unbinding rule.

## C / strongest counter-hypothesis
At only eight tasks, address metadata erases any superposition savings; the chosen random PSP codebook and direct unbinding may be a weak PSP implementation.

## U / limits
Synthetic linear functions only; no nonlinear task merges, pretrained models, learned decoders, or throughput. The logged off-diagonal Gram statistic measures intrinsic task correlation rather than decoder-only interference.
