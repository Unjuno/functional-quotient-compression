# MA-494 — Error-correcting Mirror expert IDs

Status: **PROMISING, scoped synthetic address-channel result**.

**H:** Redundant expert IDs reduce wrong dispatch under bit corruption at a modest serialized codebook cost.

**T:** 32 logical experts; binary 5-bit IDs, seeded ECOC codewords of 7/9/11 bits, and 3x repeated 15-bit binary IDs. Nearest-Hamming decoder; 4,000 simulated routes per fresh world/seed/noise point; fresh worlds 49420-49422 × seeds 0-2. A0 without the registered repetition control is preserved and excluded. A1 includes 225 payload/noise result rows; codebooks and expert mapping are charged.

**D:** PROMISING for this injected-noise address channel. At p=.1, binary accuracy was .591. ECOC-11 reached .816 (misroute reduction 55%), while 3x repetition reached .868 (misroute reduction 68%). Serialized payloads were 2,149B and 2,277B versus binary 1,957B (1.10x and 1.16x), below the preregistered 2x ceiling. All methods were 100% accurate without noise. At p=.2, accuracy was .510 for ECOC-11 and .574 for repetition, showing the robustness boundary.

**C:** This is an artificial independent bit-flip channel. The repetition control performed better than the selected ECOC family, so the result supports address redundancy generally, not a Mirror-specific code advantage.

**U:** Real MoE router errors, semantic cross-talk costs, learned router noise, and task-level quality remain untested.
