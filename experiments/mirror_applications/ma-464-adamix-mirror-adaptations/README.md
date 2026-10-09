# MA-464 — AdaMix over logical Mirror adaptations

Status: **FAIL for routed mixture claim; merged outputs equivalent**
Evidence lane: STOCHASTIC ADAPTATION MIXTURE / QUALITY / MERGED INFERENCE / BYTES
Protocol frozen: `7b0839ca`; fresh worlds 46410–46412.

## H — Hypothesis

Stochastic training of four logical Mirror views over one physical 2×2 basis can match independent AdaMix adapters on fresh routed skills with substantially fewer bytes, while retaining comparable quality after single-adapter merging.

## T — Test

Four-skill 2D linear regression. Three skill matrices were output rotations of one basis; a fourth had an off-orbit residual. Compared four independent stochastic AdaMix adapters, one shared basis with four trainable Givens views, routed inference, arithmetic-mean merged inference, one shared adapter, and independent fitted upper control. Training used 1,000 balanced stochastic skill/component updates. Three fresh worlds × three seeds; actual inference packages measured for N=1 and N=4.

## D — FAIL for routed Mirror mixture

**Fact:** At N=4, routed AdaMix achieved mean NRMSE 3.04e-6 at 1,957B total; routed Mirror achieved 0.2235 at 2,209B. Mirror is both less accurate and larger. Merged AdaMix had NRMSE 0.5931 and 1,705B; merged Mirror had 0.5966 and the same 1,705B. Shared-only had 0.5962 at 1,641B. Independent upper control was exact at 1,957B.

**Interpretation:** Mirror views do not recover the independent mixture's routed skill functions in this mixed aligned/off-orbit case. Merging collapses skill diversity for both methods; Mirror's merged model is essentially equal to AdaMix's merged model and slightly worse than the shared-only baseline, with no byte advantage.

## C — Strongest counter-hypothesis

Three skills lie on a shared rotation orbit but the fourth requires private residual capacity. The shared Mirror basis cannot represent this off-orbit skill, while AdaMix's independent modules can.

## U — Unknown

Larger adapters, learned routers, natural language task mixtures, and partial private residuals remain untested. No Transformer evidence.
