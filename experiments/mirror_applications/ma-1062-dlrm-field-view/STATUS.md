# MA-1062 status

Phase: development complete; AM-01 pre-audit correction is being frozen. The official `ua.test` ratings remain unopened.

Development uses 81,490 training / 9,080 validation ratings from `ua.base`, split by deterministic user/movie hash. Each method selected epoch independently from 3/5/8 using mean validation logloss across seeds 106200 and 106201. H=512 Mirror improved over paired diagonal by only about 0.00044 logloss, below the 0.002 gate. Pre-audit review added the omitted per-coordinate diagonal comparator and ensures target labels/IDs will not be persisted. The fresh audit settings, seeds and corrected runner are frozen before evaluation.
