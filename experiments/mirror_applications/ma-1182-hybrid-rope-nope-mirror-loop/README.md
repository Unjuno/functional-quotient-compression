# MA-1182: RoPE/NoPE Hybrid × Mirror activation × recurrent Attention

**Status: PRE_REGISTERED, not trained.** This is a new preregistered follow-up after MA-1181, not a changed threshold or reanalysis. Strictly separate from MA-1178/1179/1180 (which had no RoPE) and MA-1181 (NoPE-only full positional swap, screening FAIL).

## Question
With a shared two-pass Transformer block, can a learned two-code Householder nonlinear value transform on one head whose attention uses NoPE improve a hybrid architecture (other head uses RoPE) compared with equal-code FiLM and nearly-equal-code *coupled rank1* at matched steps? Does it preserve separately measured order-sensitive capabilities?

The implementation uses 2 heads (d_head16), d_model32, SiLU FFN width64, no MoE. All heads share the same recurrent block weights across two passes; one head uses position-aware RoPE when hybrid, the second head NoPE. Crucially, Mirror is placed on the second head's **value activation**, not as an identical Q/K rotation that cancels algebraically in NoPE. Base attention values in **all** arms use SiLU so the nonlinear pathway is controlled.

## Strong non-Mirror control
The coupled-rank1 transformation `S=I+beta*u*u^T`, used on both sides of SiLU, reproduces Mirror identically at beta=-2. It has only two more coefficients than Mirror and is initialized with the same u directions and beta=-2. This strong comparator isolates whether hard orthogonality is beneficial or restrictive. A Mirror victory vs FiLM alone is NOT a victory vs rank1.

## Domains and evaluation
3 tasks: (i) query-key membership independent of order, (ii) latest content's bit (position), (iii) most recent occurrence of query key (mixed semantic/position). Balanced labels, no oracle task IDs given to model except query content. Training/DEV/Fresh use 10 content tokens plus query; OOD uses 14 content tokens plus query.

Compare `nope`, `rope`, `hybrid` across `shared`, `film`, `mirror`, `coupled_rank1`, `fixed_mirror`: 3 tasks ×3 seeds ×3 positions ×5 arms = 135 independent training trajectories. 450 AdamW updates, 64 batch each, checkpoint at 225 and 450 chosen by DEV; no sweep using Fresh. Freeze checkpoint hashes and code before Fresh/OOD. See `PROTOCOL.json` for **predeclared primary gate**.

## Resource and scope
Physical parameter counts, actual `torch.save` inference bytes, update counts, CPU one-thread wall time, and batch64 forward medians recorded separately. Same number of updates is NOT FLOP equality; special function costs and extra rank1 operations are not free. Synthetic exploratory screening only, no LM benchmark or long-context claims. Preserve FAIL and do not merge research branch to main.