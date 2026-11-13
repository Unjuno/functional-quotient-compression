# MA-903 results — rigid SE(3) mechanism screen

## H — hypothesis

One canonical point cloud plus compact SE(3) codes reconstructs unseen action compositions with fewer serialized bytes than materialized posed point clouds. Mirror-specific value must improve on ordinary native SE(3) parameters with the same decoder.

## T — execution

Four synthetic canonical point clouds (256 points each), six seeded rigid action transforms, and all 36 ordered action pairs per object were evaluated. Development seeds were 90301/90307; amended fresh seeds 90331/90333/90339. Every composition was decoded from the two action matrices. Compared canonical geometry plus primitive SE(3) codes (Mirror), the identical native SE(3) state/decoder, and fully materialized posed point clouds. All inference states were serialized as compressed NPZ and reopened.

## D — FAIL for Mirror-specific attribution; synthetic storage mechanism passes

### Facts

| Fresh seed | Mirror payload | Native SE(3) payload | Independent posed cloud bank | Byte reduction | Mirror/native max point error | Mirror transform MAC proxy |
|---|---:|---:|---:|---:|---:|---:|
| 90331 | 12,348 B | 12,348 B | 409,738 B | 96.99% | 0 / 0 | 1,728 |
| 90333 | 12,363 B | 12,363 B | 409,826 B | 96.98% | 0 / 0 | 1,728 |
| 90339 | 12,372 B | 12,372 B | 410,119 B | 96.98% | 0 / 0 | 1,728 |

The transforms passed rotation orthogonality, positive determinant, and homogeneous-coordinate checks. Reopened payloads replayed with exactly zero point error. Mirror and native are byte-identical and use the same transform compute.

### Interpretation

Canonical geometry plus six rigid action transforms compactly represents the 144 object/action-pair views in this synthetic setup, saving about 97% actual payload bytes versus materializing every posed cloud. This benefit is exactly ordinary SE(3) parameter sharing and composition; it is not a Mirror-specific gain. Materialized-cloud lookup also avoids the transform compute required by shared decoding.

### C — strongest counter-hypothesis

The compression is standard rigid-body instancing. It relies on known action transforms and rigid geometry, not learned action-conditioned world dynamics. Nonrigid motion, appearance, occlusion, and interaction may require private state that removes the savings.

### U — unresolved

PA252's learned MRO-GWM dynamics, real object scans, action-conditioned rollout error, camera render quality, occlusion, and nonrigid/private-residual frontier remain untested.

## Protocol correction retained

The first implementation charged the native control for a table of all composed pose matrices while Mirror stored primitive transforms. This was unfair. Those outputs are preserved under `source/results/pre_amendment/` and excluded. Amendment 01 makes the native state exactly the same six primitive SE(3) transforms/decoder and uses fresh seeds 90331/90333/90339. No tuning used the initial fresh outputs.
