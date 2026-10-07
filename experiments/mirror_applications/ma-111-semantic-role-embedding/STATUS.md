# MA-111 status

- Status: SCREENING
- Branch: `research/ma-111-semantic-role-embedding-20261007`
- Base commit: `6e0ed10a094f9a082a144a2d8d86f508fd4cbad6`
- Protocol frozen: yes, before development in the current branch commit.
- Development complete: no
- Fresh/audit opened: no; 11111–11113 sealed.
- Results committed: no
- Verification committed: no

## H — hypothesis

A shared input embedding plus one Givens coordinate per semantic role preserves held-out-filler next-token distributions at lower actual bytes than private role maps; unrelated role transforms need private capacity.

## T — setup

Four role IDs, 32 filler tokens, shared 16D embeddings and 8-way readout. Compare hard tie, role addition, FiLM, rank-2 input/output LoRA, a full per-role output head, fixed VSA sign binding, full role maps, and Mirror Givens views. Development seeds 11101/11102, LR .003/.01; fresh 11111–11113 remain sealed.

## D — pending

PA13, TPR/filler-role representation, tied-embedding analysis, Kronecker input embeddings and shared semantic-concept embeddings were reviewed. Controls and thresholds are drafted; implementation and protocol freeze are pending.

## C — strongest counter-hypothesis

A rank-2 output-head LoRA can represent the linear effect of a Givens input view and may fit as well with a broader usable function family.

## U — unresolved

No results. Semantic-role names are controlled synthetic role IDs; natural semantic binding and language-model NLL remain untested.
