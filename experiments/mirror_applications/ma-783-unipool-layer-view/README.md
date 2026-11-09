# MA-783 — UniPool global expert pool with layer role views

Status: **FAIL** for Mirror-specific value (synthetic routed operator screen).

## Fact

Four global 8×8 experts plus four per-layer routers use 2,337 B but yield routed output nRMSE .2909 when a layer Givens coordinate is omitted. Adding four Givens angles per layer gives 2,590 B and nRMSE `1.00e-7`, compared with a 5,158 B independent layer-expert bank at zero error. The view uses 128 router-plus-expert MACs per token/layer vs 96 for independent experts, a 33% increase. On this CPU the timed 256×4 pass was ~3.64 ms vs ~1.74 ms independent and ~1.65 ms pooled. Mirror and native direct-angle Givens controls have identical payload hash, bytes, output and MACs.

## Interpretation

Layer coordinate codes restore the synthetic layer-specific behavior from a global pool at about half the independent payload. This demonstrates that the selected structured views preserve the designed functions. The same function and state are ordinary Givens/OFT coordinates, and applying the rotations increases compute and measured CPU time. The Mirror-specific claim fails.

## H / T / D / C / U

- **H:** Small layer views can recover useful logical specialization from a global expert pool with fewer physical expert bytes.
- **T:** Seed 78301; four layers, four global experts, dimension 8, fixed per-layer routers, 256 routed examples; controls are independent layer experts, UniPool without Views, Mirror Givens views, and native direct-angle Givens. Actual compressed NPZ bytes, routed output nRMSE, MAC proxy and CPU wall time measured; zero optimizer updates.
- **D:** **FAIL** for Mirror-specific value. Bytes fall 49.8% vs independent and quality is preserved, but MAC rises 33%, CPU pass time is ~1.9×, and native direct-angle Givens exactly aliases the View.
- **C:** This is ordinary orthogonal coordinate adaptation around each shared expert; no new Mirror function family is needed.
- **U:** Trained UniPool with learned routers, natural MoE tasks/NLL, GPU fused Givens runtime, router balance under View conditioning, and private fallback for heterogeneous layers.

## Scope

Synthetic linear operators and fixed random routers only. This does not reproduce the full UniPool training system or establish a language model, accelerator, or task-quality gain. Fresh data remained sealed after the direct-angle control exactly aliased the Mirror payload and function.
