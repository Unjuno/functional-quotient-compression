# MA-330 status

- Status: SCREENING
- Branch: `research/ma-330-tensorized-kv-reconstruction-20261009`
- Base commit: `07fee52c`
- Development complete: yes (worlds 33000–33001 × seeds 0–2)
- Fresh/audit opened: no
- Protocol locked: after this commit, before fresh

## Next action

Run frozen fresh worlds 33010–33012.

## Development finding

Aligned cache views recover attention context at far fewer bytes than full/Tucker state; hard MLKV equality loses quality. Independent roles need private cache state. Encode and materialization costs are measured separately.
