# MA-481 status

- Status: **FAIL**
- Branch: `research/ma-481-vq-mirror-logical-functions-20261008`
- Base commit: `58ed65ebb5395d36b3b5cb55929ea2eca4cad232`
- Protocol frozen: yes (`freeze.json`; amendment 1 is accounting-only)
- Development complete: yes (48101, 48102)
- Fresh/audit opened: no (48111–48113 sealed)
- Verification: byte/hash replay and native VQ alias checked; tests 3 passed

## Next action

Proceed to MA-482 residual VQ on its own research branch.

## Decisions / rulings

VQ16/64/128 all missed both the heldout RMSE <=0.05 and >=90% distinct heldout-address gates in both seeds. Int8 reached RMSE <=0.00157 at 6,101 B. Mirror/native VQ payloads and outputs were exact aliases.
