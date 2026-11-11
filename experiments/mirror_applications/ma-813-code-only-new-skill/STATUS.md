# MA-813 status

- Status: FAIL for one-skill full-byte frontier and Mirror attribution.
- Branch: `research/ma-813-code-only-new-skill-20261111`
- Base: `bb78d082`
- Development: synthetic seed 81301; rank-2 frozen basis; 32 support and 512 test examples.
- Fresh/audit: sealed after byte gate and native coefficient alias failures.
- Verification: focused test passed; serialized reconstruction replay exact.

## H / T / D / C / U

- **H:** A new skill can be added using only a small code, beating fresh LoRA on total bytes and adaptation work.
- **T:** One synthetic 8D skill, frozen two-atom basis, same 32-example support for Mirror and rank-2 LoRA.
- **D:** **FAIL.** Mirror reaches nRMSE 2.87e-8 but uses 1,440 B versus 981 B fresh LoRA and 754 B independent matrix; native shared-basis code is identical at 1,456 B.
- **C:** Basis deployment overhead overwhelms one new code; normal low-rank LoRA is cheaper.
- **U:** Multiple-skill amortization, learned/natural bases, robot data and fresh seeds.
