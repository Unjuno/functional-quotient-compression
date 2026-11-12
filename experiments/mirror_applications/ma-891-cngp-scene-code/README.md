# MA-891 — C-NGP shared scene field plus Mirror scene code

Status: **UNTESTED — execution blocker**
Branch: `research/ma-891-cngp-scene-code-20261112`
Registry status remains `UNTESTED`; this is not a scientific FAIL.

## H — proposed hypothesis

A compact scene-specific View coordinate on the shared C-NGP physical field may recover scene-specific held-out view quality and retention with fewer actual serialized bytes than native C-NGP scene conditioning plus replay state.

## T — feasibility review only

No model training, image scoring, or fresh-data access occurred. The paper-linked implementation requires Instant-NGP CUDA components (`tinycudann`, a compiled CUDA extension, and CUDA device placement); this runtime reports `torch 2.14.1+cpu`, `torch.cuda.is_available() == False`, and lacks `tinycudann`, `nerfacc`, and an Instant-NGP installation. The public C-NGP README says checkpoints were not released. No C-NGP checkpoint or scene dataset was found in the workspace.

The registered control is C-NGP's shared multi-scene field with native scene labels and generative replay. A CPU substitute using a different neural field or synthetic 2D scenes would not test that control or its radiance-field quality/storage/forgetting frontier, so no model result is claimed.

## D — status

**NOT ESTABLISHED / UNTESTED.** Hardware and model assets required for a faithful comparison are unavailable in this container. See `EXECUTION_BLOCKER.md` for evidence and next requirements.
