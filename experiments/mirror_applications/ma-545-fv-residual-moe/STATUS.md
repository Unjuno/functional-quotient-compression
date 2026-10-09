# MA-545 status

- Status: FAIL
- Branch: `research/ma-545-fv-residual-moe-20261009`
- Base commit: `a37cb993d0a9bef5c7a9bb39e86abca39ed30bf6`
- Development complete: yes
- Fresh seeds 54511–54513: complete; settings stayed frozen
- Results committed: yes (`0a520e4ab65f42f7cebf4f7b444b1922e97ffdaf`)
- Verification committed: yes
- Registry row updated: yes; FAIL

## Next action

Commit the integrated FAIL result and start MA-546 on its dedicated branch.

## Blockers

None.

## Decision

FACT: Router accuracy and FV-vs-oracle quality pass, but shared-mean intervention beats routed FV candidate accuracy in all five seeds at one-eighth the fresh payload. Routed FV has a secondary gold-logprob advantage.
INTERPRETATION: Few-shot-context routing is easy here; per-task FV experts do not improve the primary answer-accuracy/storage point.
HYPOTHESIS: FV routing may be useful when calibrated likelihood is the objective; this remains untested.
