# MA-655 — Stored-program memory with Mirror program codes

Status: FAIL for Mirror-specific advantage
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: 28194ea (research/mirror-application-worker-ready-20261007)

## H — hypothesis

A small task program coordinate can express held-out interpolated controller functions using a shared controller basis and fewer bytes than an NSPM-style key-value memory of full anchor programs, while matching trajectory quality and switch cost.

## Mirror insertion

> **Mirror insertion:** this experiment adds per-task coordinate m=(z1,z2) to a shared recurrent controller basis so that a task-specific controller can be reconstructed without storing a full controller program for every task.

The physical object is a controller matrix program. Each task coordinate is persistent, serialized and charged. One unrelated task has a private full program. The nearest ordinary control is a shared low-rank matrix basis with per-task coefficients; it is algebraically equivalent to the proposed Mirror code here. The test is intended to detect that equivalence, not to infer a Mirror-specific benefit from it.

## Prior-art delta

PA134's Neural Stored-program Memory stores controller-weight programs in key-value memory and retrieves/interpolates them dynamically. This mechanism screen compares four-corner full program memory, Mirror affine coordinates, an ordinary shared matrix-basis coefficient control, and independent full programs. Program interpolation on held-out task coordinates is the primary quality check. This is not a reproduction of the full NSPM controller or learned key-query network. Basis matrices and codes are supplied by the synthetic teacher; there is no gradient fitting.

## Task

A 12-dimensional recurrent controller updates state for 8 steps using A(z)=A0+z1*B1+z2*B2 for 23 aligned tasks inside [-1,1]^2; one unrelated task uses a private matrix. The NSPM-style baseline stores four corner programs and bilinearly interpolates their weights. Mirror stores A0/B1/B2 and task codes. Both receive and pay the same task coordinates. Held-out coordinates are sampled inside the convex hull; one private task checks the fallback boundary.

## Controls

1. NSPM-style four-corner key-value program memory plus private program.
2. Mirror shared affine controller basis plus per-task code and private program.
3. Ordinary low-rank shared matrix basis plus coefficients (same expressive class as Mirror).
4. Independent full controller programs.

All methods are serialized from actual inference payloads. Report normalized state-trajectory MSE, exact payload bytes, controller MAC proxy, and CPU task-switch wall time.

## Gates

**PASS:** Mirror matches fresh held-out trajectory MSE <=1e-5, uses at least 10% fewer bytes than NSPM-style memory, and has switch MACs <=1.25x NSPM.

**Mirror-specific pass:** in addition, Mirror must use at least 10% fewer bytes than the ordinary shared-basis coefficient control at matched quality.

**FAIL:** quality or NSPM byte/runtime gate misses, or the ordinary shared-basis control matches within 10%. The latter means standard program coefficients suffice; it does not invalidate the broad shared-code mechanism.

## Selection

Random draw 9 selected MA-655 from 454 eligible P0/UNTESTED IDs at index 205. Pool list is saved in source/selection_pool.csv and hashes are recorded in PROTOCOL.json. No MA-655 branch was found.

## H / T / D / C / U

**H:** A compact per-task program coordinate may recover held-out controllers with less state than full program memory.

**T:** 12-D controller, 8 steps, 23 affine tasks plus one private task, 32 held-out codes per world, fresh seeds 65511–65513. Compared four-corner NSPM-style memory, Mirror basis+code, ordinary shared basis, and independent programs.

**D:** Mirror-specific FAIL; broad mechanism passes against NSPM-style memory.

**C:** Ordinary shared matrix coefficients are algebraically the same model and match Mirror quality with 2 B more payload.

**U:** Neural query/key retrieval, nonlinear programs, learned codes, long-horizon stability, and real serving latency.

## Fact / Interpretation / Hypothesis

FACT: On all 3 fresh worlds, Mirror and ordinary shared-basis control have normalized trajectory MSE <4e-15. Mirror uses 2,837 B vs 3,327 B for NSPM-style corner memory (14.7% fewer) and 2,839 B for the ordinary shared-basis control; controller MACs are 117,888 vs 150,656 NSPM.
INTERPRETATION: Three stored basis programs plus per-task coordinates compress the four-corner program memory, but the ordinary low-rank matrix basis reproduces the same function and costs effectively the same bytes/compute. The Mirror-specific gate fails.
HYPOTHESIS: In this affine task family, Mirror codes are an equivalent notation for ordinary program coefficients; a benefit beyond that requires non-affine behavior or more efficient code extraction.
BOUNDARY: Synthetic recurrent linear controller family only; supplied code coordinates, no learned memory/query network.
