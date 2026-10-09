# MA-296 — Orthogonalized task-vector Mirror superposition

Status: FAIL. Base: `66f551ec`. Prior art: PA16 Parameter Superposition and PA26 Task Arithmetic.

## H / hypothesis

A task address can retrieve correlated task deltas from a shared superposition with lower interference and smaller serialized payload than independent task vectors. Generic orthogonal task bases and PSP may explain any gain.

## T / protocol

Synthetic linear task edits (D=512, N=8), 640 support and 128 query examples per task. Development worlds 29600–29601, seeds 0–2; fresh worlds 29610–29612, seeds 0–2. Compare unbound sum, independent vectors, random-context PSP, fixed Hadamard address, and generic QR basis. Address/basis metadata is charged in actual serialized payload. This screens task-vector retrieval and interference only; it is not an LM capacity test.

## D / decision

FAIL on fresh worlds: Hadamard and generic QR reconstruct at about 2e-6 query NRMSE but use 16,735 B and 16,744 B versus 16,473 B raw vectors. Unbound sum is 2,131 B but query NRMSE is about 2.52. Random PSP codes score about 0.94 under the fixed unbinding rule. No Mirror-specific storage or quality gain.

## C / strongest counter-hypothesis

The address table adds bytes without reducing the N×D task state; any exact retrieval is ordinary orthogonal coding rather than a Mirror-specific effect.

## U / limits

Synthetic linear functions; no nonlinear fine-tuning, real pretrained tasks, or inference-kernel benchmark.

Development screen: address-bound Hadamard and generic QR reconstruction both reach approximately 2e-6 NRMSE, matching raw independent estimates, but serialize slightly more bytes than raw vectors. Random nonorthogonal PSP codes show substantial crosstalk. The unbound sum is smallest but has high error/interference. Fresh worlds test whether this pattern generalizes; no tuning is planned.
