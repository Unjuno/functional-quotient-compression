# MA-860 — new-task projection into a learned subspace

Learn a rank-3 task-delta basis from three rotated Digits source tasks, then adapt to a held-out fourth rotation using 32 labeled support examples. Compare against random rank-3 and full fine-tuning, and charge the shared base, all basis vectors, and task code by actual serialized bytes. Protocol is frozen in PROTOCOL.json.
