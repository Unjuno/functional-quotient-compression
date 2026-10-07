# MA-199 status

- Status: SCREENING
- Branch: `research/ma-199-gradient-coordinate-20261007`
- Base commit: `a5dc84c59a6a7fd35c43dc40003f3be52cc03d75`
- Protocol frozen: no
- Development complete: no
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: no

## H — hypothesis

A task-specific angular coordinate in a paid shared gradient plane can encode aligned rank-1 skill updates with fewer actual bytes than private rank-1 LoRA, while unrelated updates need private capacity.

## T — fixed setup

16x8 base, shared 16x2 plane, four sequential tasks. Compare angular coordinates plus one update vector to generic plane coefficients, rank-1/2 LoRA, hard tie and independent full maps. Dev seeds 19901/19902; fresh seeds 19911–19913 remain locked.

## D — pending

Targeted prior-art review is complete. Protocol and implementation are in progress.

## C — strongest counter-hypothesis

The ordinary shared-plane coefficient matrix may fit the same functions at comparable bytes, making the angular Mirror code an unnecessary constraint.

## U — unresolved

No measurements yet. Basis discovery is not charged in this screen; only the supplied basis and its ongoing state are counted.
