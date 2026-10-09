# MA-478 status

- Status: FAIL for frozen <=80% storage gate; private fallback boundary is established
- Branch: `research/ma-478-view-first-private-fallback-20261009`
- Base commit: `fb8567f7`
- Protocol freeze: `fe2ffb3c`
- Development complete: yes; selected tau=0.30
- Fresh/audit opened: yes, after threshold freeze
- Results committed: yes (28397956)
- Verification committed: yes (28397956)
- Registry row updated: yes (28397956)

## Next action

MA-478 is complete; continue with MA-481 on its dedicated branch.

## Limitations

- The known 70/30 alignment mixture is synthetic; no learned task distribution was used.
- Fallback stores full private output vectors. Residual or quantized fallback was not tested.
- The registered <=80% storage gate fails at all amortization sizes.

## Decisions / rulings

- Development sweep chose tau=0.30 by the preregistered highest-threshold rule; fresh worlds were not used for threshold selection.
- At N=64 the 19/64 private entries match the generated off-orbit fraction; all keys, radii, flags, indices, basis, codes and private vectors were charged.
- Oracle-labeled fallback is a diagnostic upper control and matches the residual-threshold policy on this separable synthetic mixture.
