# MA-569 status

- Status: FAIL (Mirror-specific gate)
- Branch: `research/ma-569-universal-transformer-step-mirror-20261009`
- Base commit: `dc505c56`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no

## Next action

Update registry and claim ledger, replay verifier, and commit the development-only FAIL.

## Blockers

None.

## Decisions / rulings

Two seeds: Mirror step angle and direct time-angle code both use 1207B and exactly recover odd held-out step functions. Untied matrices use 2510B at zero error; hard tying uses 716B at nMSE .16679; oracle rank-1 LoRA uses 1192B at nMSE .10727. This is aligned linear recurrence only; fresh remains sealed because the direct code aliases Mirror.

## Decisions / rulings

Two seeds: Mirror step angle and direct time-angle code both use 1207B and exactly recover odd held-out step functions. Untied matrices use 2510B at zero error; hard tying uses 716B at nMSE .16679; oracle rank-1 LoRA uses 1192B at nMSE .10727. This is aligned linear recurrence only; fresh remains sealed because the direct code aliases Mirror.
