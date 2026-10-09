# MA-586 status

- Status: FAIL (Mirror-specific/frontier gate)
- Branch: `research/ma-586-quantized-mla-mirror-gauge-20261009`
- Base commit: `99351d0b`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no

## Next action

Update registry/claim ledger, replay verification, and commit the development-only FAIL.

## Blockers

None.

## Decisions / rulings

Angle selection used the first 32 queries and evaluation used held-out 32 queries. Mean attention-output nMSE: identity .06063, optimized Givens .03811, Hadamard .01943. Payloads: identity/Hadamard 650B, Givens 647B. Direct Givens matched Mirror exactly in bytes and quality. The optimized gauge beats identity but loses to the stronger Hadamard control; no Mirror-specific or overall frontier gain. Fresh remains sealed.
