# MA-1062 status

Phase: development complete; fresh protocol/settings/seeds are frozen in `FRESH_FREEZE.json`. The official `ua.test` ratings were not read during development.

Development uses 81,490 training / 9,080 validation ratings from `ua.base`, split by deterministic user/movie hash. Each method selected epoch independently from 3/5/8 using mean validation logloss across seeds 106200 and 106201. Mirror H=512 improved over paired diagonal by only about 0.00044 logloss, below the frozen 0.002 gate. The official fresh audit remains necessary to complete this scoped candidate.
