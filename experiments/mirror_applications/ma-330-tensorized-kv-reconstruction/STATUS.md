# MA-330 status

- Status: PROMISING (narrow aligned synthetic result)
- Branch: `research/ma-330-tensorized-kv-reconstruction-20261009`
- Base commit: `07fee52c`
- Development complete: yes (worlds 33000–33001 × seeds 0–2)
- Fresh/audit opened: yes; worlds 33010–33012 × seeds 0–2
- Protocol locked: commit 61760a6b, before fresh

## Decision

The aligned stratum clears the context quality threshold and beats rank-2 PCA/Tucker bytes by 64.8%; hard sharing fails quality. Independent states fail for Mirror and PCA/Tucker, confirming the need for private state off-orbit. Record as PROMISING only for the constructed Givens orbit. Materialization is materially slower than PCA/Tucker. No natural KV or language-model claim.

## Development finding

Aligned cache views recover attention context at far fewer bytes than full/Tucker state; hard MLKV equality loses quality. Independent roles need private cache state. Encode and materialization costs are measured separately.
