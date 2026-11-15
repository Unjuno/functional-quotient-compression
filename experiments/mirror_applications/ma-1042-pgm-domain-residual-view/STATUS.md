# MA-1042 status

- Status: fresh audit in progress
- Branch: `research/ma-1042-pgm-domain-residual-view-20261115`
- Official PGM-index source: pinned commit `c6fcf3d34e55eb0061b01e2f49dfcbdb711f1407`; Apache-2.0
- Native DynamicPGMIndex and PGMIndex compiled with g++17 in the available CPU container
- Protocol and both API amendments frozen before metric execution
- Development epsilon frozen at 8; all candidates tied on official DynamicPGMIndex RAM, so protocol tie-break chooses smaller epsilon
- Fresh audit seeds 1044–1046 remain untouched until `runs/DEV_FREEZE.json` was written

## Current implementation

Official DynamicPGMIndex handles the initial key set and ordered insert/delete stream. Because its iterator API is forward-only, post-update predecessor queries use an official static PGMIndex deterministically rebuilt from the final keys. The program checks all updates through official dynamic-index membership and exact predecessor outputs against the ordered-vector oracle. The matched native affine-coordinate control executes the same inverse coordinate map as the Mirror-labelled path.

## Next action

Run the frozen epsilon 8 configuration on fresh seeds 1044, 1045 and 1046, retaining actual serialized payloads, compute, exactness and runtime outputs.
