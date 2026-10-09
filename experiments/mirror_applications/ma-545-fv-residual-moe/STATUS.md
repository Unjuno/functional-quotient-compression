# MA-545 status

- Status: SCREENING
- Branch: `research/ma-545-fv-residual-moe-20261009`
- Base commit: `a37cb993d0a9bef5c7a9bb39e86abca39ed30bf6`
- Protocol frozen: yes
- Development complete: yes
- Fresh/audit opened: yes (54511–54513 opened after committed gate)
- Results committed: partially (fresh 54511)
- Verification committed: partial
- Registry row updated: pending

## Next action

Run frozen fresh seeds 54511, 54512 and 54513 without tuning; 54511–54512 complete; 54513 running.

## Blockers

None.

## Amendment 1

Fixed NumPy/PyTorch RNG to the world seed because the original freeze omitted initialization determinism. Original dev outputs preserved in `results/pre_amendment_1/`. Fresh data remains sealed; both development seeds are being rerun under the amended freeze.
