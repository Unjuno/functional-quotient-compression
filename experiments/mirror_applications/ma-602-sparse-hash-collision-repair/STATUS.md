# MA-602 status

- Status: **FAIL (development, verified)**
- Branch: `research/ma-602-sparse-hash-collision-repair-20261009`
- Dev seeds 60201/60202 complete; fresh 60211–60213 sealed.
- 6,181/6,180 private exceptions; collision+Givens 34,995/34,991 B vs native hash4096 13,791 B; accuracy .9822/.9800.
- Verification passed: 16/16 payloads replay byte-exact; 3 tests passed.

## H / T / D / C / U

- **H:** Private collision exceptions plus Givens recover native hash4096 quality at no greater bytes and beat random exceptions.
- **T:** Digits MLP, 8 methods, 800 updates, two dev seeds; serialized table/index/residual payloads measured.
- **D:** **FAIL.** Required +1pp gains fail; all duplicate exceptions produce 75.4% private coverage and 2.54x hash4096 bytes. Random placement is as good or better; no-view collision residual is within .5pp of Mirror.
- **C:** 2,048-bucket collision density makes this exception scheme non-sparse; larger native hash is more byte-efficient.
- **U:** No selective task-harmful exception scoring, large model or language-model result.

## Facts / interpretation / hypothesis

- **Fact:** Two dev worlds show no qualifying quality gain and large payload expansion.
- **Interpretation:** Private parameters become necessary only for some collisions, but this uniform duplicate-repair rule marks too many.
- **Hypothesis:** A much smaller gradient- or task-salience-selected exception set may preserve the storage frontier; that would require a new preregistered candidate.
