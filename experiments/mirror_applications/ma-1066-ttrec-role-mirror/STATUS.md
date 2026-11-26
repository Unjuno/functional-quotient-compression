# MA-1066 status

Phase: SCREENING preparation. The protocol, runner, controls, audit-freeze utility, and upstream dependency are pinned; 3 CPU view tests pass. GPU Control registration for source commit `ac5957c0fc1cb497e66e7e1be05ef07419171cf8` (`cd662491288542b08c03bf103d9bf8ec`) is pending review and not executable; preview returned `TRAINING_PROFILE_UNAVAILABLE`. The service only executes its configured MAP-44M pinned commit and has no public profile-provisioning operation. The worker has CPU-only PyTorch and no CUDA device. No training, fresh audit, payload measurement, or QPS measurement has been performed.

The hypothesis remains untested. Do not mark PASS/FAIL from CPU-only adapter checks. Reopen by reviewing the committed runner through GPU Control, confirming a usable GPU and cost ceiling, then freezing source/protocol hashes before any audit access.

Prior-art reference: PA340, TT-Rec / TT-EmbeddingBag. Upstream FBTT-Embedding commit: `b95947ccd32cf9ab851242cc606562f2a74d9efa`.
