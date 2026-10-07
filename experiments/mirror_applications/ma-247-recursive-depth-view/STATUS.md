# MA-247 status

- Status: **FAIL (development screen)**
- Branch: `research/ma-247-recursive-depth-view-20261007`
- Base commit: `f11fb3cf08634d1a79c1b7c7366bad49b8828ff2`
- Development complete: yes
- Fresh/audit opened: no; unused seeds 24701–24703
- Results committed: pending
- Verification committed: pending
- Registry row updated: pending

## Decision

At selected common LR 0.01, Mirror missed the preregistered quality screen on development world 24700: MSE 2.664e-3 versus 2.012e-3 tied, 6.384e-4 scalar gate, and 1.076e-3 static LoRA. It used 3,297 serialized bytes, 46.7% below untied, but its quality gate failed. At LR 0.003 it was worst of six methods. Per protocol, do not open fresh worlds for this failed screen.

## Verification

The 12 development rows were deterministically replayed from the frozen source and split. All serialized-byte counts matched exactly; maximum MSE delta was 4.51e-13. Test results are in VERIFICATION.json.

## Blockers

None. This is a scoped negative result, not a hardware blocker.
