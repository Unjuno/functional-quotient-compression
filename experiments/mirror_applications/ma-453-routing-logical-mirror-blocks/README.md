# MA-453 — Routing Networks with logical Mirror blocks

## H — Hypothesis

One shared nonlinear block with two continuous Mirror coordinates can expand the logical block vocabulary beyond finite prototype routing without increasing bytes over independent task blocks.

## T — Test

Fresh scalar regression tasks y=tanh(a*x+b) across three worlds, three seeds, and 64 tasks/seed. Development selected K=8 prototype blocks from K=4/8. Compare shared-only, Routing Network prototypes, shared tanh plus learned Mirror scale/shift, and independent per-task scale/shift. Actual N=1/20/64 packages were serialized.

## D — FAIL for compression claim; scoped routing-frontier PROMISING

At N=20, Mirror NRMSE 0.0022 beats Routing K=8 0.0837; payload is 91.65B/task vs Routing 97.85B/task. But Mirror exactly matches independent block quality and bytes (0.0022, 91.65B/task). There is no storage gain over the simpler two-parameter private block.

## C — Strongest counter-hypothesis

The Mirror code is the same two scalar weights as the independent block; the activation-only shared module does not reduce task state.

## U — Unknown

No large-model or natural-task evidence.
