# MA-355 status

- Status: FAIL for Mirror-specific value; narrow oracle factorized-basis result
- Branch: `research/ma-355-product-key-mirror-address-20261008`
- Base commit: `c935a90`
- Last verified commit: `14e94ae5230afbdb7312f988c689457ec9cef4f7`
- Development complete: yes; one invalid attempt excluded by pre-fresh amendment
- Fresh/audit opened: no; seeds 35511–35513 remain sealed
- Results committed: yes
- Verification committed: yes
- Registry row updated: yes on integration branch (FAIL)

## Decision summary

Factorized state used ~5.1KB with zero error; flat/Product-Key used ~30.6–31.0KB. Direct coefficient control was only 12B larger, so no Mirror-specific value. Fresh remained sealed.

## Next action

Integrated from dedicated branch commit `14e94ae5230afbdb7312f988c689457ec9cef4f7`; 10 amended rows replay exactly and three tests pass. Fresh remains sealed because the direct control missed the 10% Mirror-specific margin.
