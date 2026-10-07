# MA-245 status

- Status: SCREENING
- Branch: `research/ma-245-mlkv-layer-views-20261007`
- Base commit: `1c313c1ac6aba2b5c4cd4bcc3933c08b1f98ce19`
- Last verified commit: pending preregistration commit
- Development complete: yes; selected common LR `0.003`
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

Freeze source/protocol and run fresh worlds 24501–24503 at 900 updates.

## Blockers

None. CPU PyTorch is available; GPU is not needed for this mechanism screen.

## Decisions / rulings

- PA08's main comparison is cross-layer cache sharing with explicit layer groups. The screen includes four independent layers, two MLKV groups, one hard-shared group, and layer-specific views.
- The teacher is exactly a shared K/V pair under layer/role views, an optimistic feasibility setup. Every layer uses the same memory hidden state to isolate projection and cache sharing; evolving layer-state interactions are not modeled.

- Development selected common LR `0.003` from the two-value grid. The view passes quality and gate-control checks in development; actual payload reduction versus two-group MLKV is 11.3%, below the preregistered 25% threshold. Cache tensor audit shows 256 bytes for one group versus 512 bytes for MLKV-2.
