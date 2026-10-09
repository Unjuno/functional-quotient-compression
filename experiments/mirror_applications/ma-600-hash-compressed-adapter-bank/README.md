# MA-600 — Hash-compressed task adapter bank

Status: **FAIL (development; fresh sealed)**. Prior art PA120 HashedNets and PA18 VeRA. This screen compares task-specific hash addressing and Givens Views against native low-rank adapter methods.

## H — Hypothesis

One shared 512-bucket table plus task-specific Givens coordinates could represent eight adapters at <=0.75x independent-hash bytes, with normalized heldout error <=.10 and within .02 absolute RMSE of independent rank-2 LoRA.

## T — Execution

Eight independent seeded rank-2 task operators were evaluated on normalized sklearn digits inputs. Methods: independent dense matrices, independent rank-2 factors, independent hash tables, shared salted hashing, shared hashing plus Givens, VeRA-style shared random factors plus task scales, and generic shared matrix basis. Each method received 1,200 AdamW updates in worlds 60001–60003. Fresh 60011–60013 stayed sealed. NPZ payloads were loaded for evaluation; all learned bases/codes/tables/metadata were serialized.

## D — Decision

**FAIL.** Mirror payload is 3,314 B versus 9,752 B for independent hash (0.340x), but aggregate NRMSE is .504/.420/.408, missing the <=.10 gate. It improves modestly over shared salted hashing (NRMSE .594/.520/.497) but does not meet the preregistered .01 absolute RMSE margin over both salted and VeRA controls, and is far worse than independent rank-2 LoRA at 4,318 B (NRMSE .0027/.0105/.0139). The View creates logical task outputs, but this task bank needs private low-rank factors for quality.

## C — Strongest counter-hypothesis

The targets were deliberately independent rank-2 operators, so low-rank per-task factors match their construction while one shared table and input rotations cannot. This is a private-parameter boundary screen, not a natural adapter benchmark.

## U — Limits

Real digit inputs are used, but the eight task targets are synthetic continuous operators. Natural PEFT tasks, language models, near-convergence capacity and GPU kernels remain untested. The reported forward throughput times task-matrix application after loading; hash expansion and View materialization are reported separately as operations, not folded into that short CPU timing.

## Facts / interpretation / hypothesis

- **Fact:** Across three worlds, Mirror used 3,314 B; independent rank-2 LoRA used 4,318 B and had much lower NRMSE. Shared salted hash used 2,586 B but had NRMSE .497–.594. VeRA random factors had NRMSE .808–.947. Fresh stayed sealed.
- **Interpretation:** Hash addressing shares adapter storage aggressively, but this independent task bank does not preserve function quality. A very small storage reduction versus rank-2 LoRA (1,004 B) does not justify the large heldout error.
- **Hypothesis:** task-specific low-rank factors carry essential private function information; learned input rotations cannot recover it from a common hashed adapter.
