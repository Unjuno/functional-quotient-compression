# MA-626 status

- Status: SCREENING
- Branch: `research/ma-626-deepsdf-mirror-autodecoder-20261103`
- Base commit: `70d55f30`
- Protocol frozen: yes
- Development complete: no
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no

## H / T / D / C / U

**H:** A canonical shared SDF decoder plus a small optimized per-shape coordinate code stores held-out functions more compactly than independent fields. A Mirror claim additionally must beat standard direct shape conditioning.

**T:** Frozen 2D translated/scaled circle family, 16 instances with four held out, 64 support and 256 audit points, development seeds 62601/62602. Fit codes from support only. Compare independent sampled SDF grids, Mirror coordinate decoder, and native direct shape conditioner. Fresh seeds are sealed.

**D:** Pending.

**C:** The Mirror coordinate is exactly the ordinary circle center/radius parameterization and may add arithmetic with no functional gain.

**U:** Learned DeepSDF decoder training and general shape classes remain outside this restricted screen.
