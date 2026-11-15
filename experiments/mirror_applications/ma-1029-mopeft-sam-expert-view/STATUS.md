# MA-1029 status

Stage: COMPLETE — FAIL, scoped synthetic SAM ViT-B feature-adapter screen.
Branch: `research/ma-1029-mopeft-sam-expert-view-20261115`.
Prior art: PA317, MoPEFT paper. Official MoPEFT code was not located in the bounded source search; the mixture control is paper-inspired, not an official reproduction.

**H:** Shared rank-8 Adapter plus two per-domain Givens coordinates should reach within 1 mIoU point of independent adapters, save at least 25% adaptation bytes, and beat matched FiLM by at least 1 point in 2/3 fresh worlds.

**T:** Frozen official SAM ViT-B; feature-space adapters on final embeddings; 3 generated visual domains; dev seed 102900; audit seeds 102901–102903; 2 train and 4 audit images/domain/world; 60 updates/method/world; zero-shot, independent, hard-shared, Givens Mirror, FiLM and paper-inspired mixture controls.

**D:** FAIL. Mirror adaptation payload is 20,505 B versus 56,959 B independent (64.0% less), and its mIoU is within 1 pp of independent in all audit worlds. It never beats FiLM by the registered 1 pp margin; FiLM scores higher in all three worlds. Hard sharing is 188 B smaller and effectively tied in quality. Full-bundle savings versus independent are only 36,454 B (0.0097%).

**C:** Near-ceiling zero-shot quality from tight box prompts on generated shapes limits detectable domain benefit; final-embedding insertion also omits MoPEFT's internal-layer design.

**U:** No natural images, MESS, official MoPEFT, other SAM backbones, encoder tuning, or GPU result. This is a small synthetic mechanism screen, not a capacity claim.

Development settings were committed before audit-data generation. Six tests and saved-result replay pass. Details: `RESULTS.md`, `VERIFICATION.json`.
