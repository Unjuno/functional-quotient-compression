# MA-792 — Routing-free expert self-activation with functional Views

Status: **FAIL** for Mirror-specific value.

## Fact

Four experts self-activate from Gaussian prototype scores on four synthetic modes. At σ=0, the Mirror view jointly generates a rank-1 expert transform and prototype positions; it reaches mode accuracy .997, balanced loads (.249–.251), and output nRMSE .0103 at 1,971 B. Independent routing-free experts use 1,866 B at the same mode accuracy/error. A central top-1 router with independent experts uses 1,659 B with the same metrics. Hard tying uses 1,124 B but output error .625. The Mirror and native low-rank output/prototype controls have identical payload hashes and metrics. At σ=.05, Mirror retains .995 routing accuracy and balanced load but output error rises to .238, versus .0346 for independent experts.

## Interpretation

Expert-local activation can retain balanced routing without a centralized router on this designed Gaussian task. Shared functional coordinates preserve the aligned output maps, but cost more bytes than the independent and central-router controls. Private expert variation breaks the shared output basis while self-activation itself remains balanced. The result does not establish a Mirror-specific routing or quality benefit.

## H / T / D / C / U

- **H:** A shared expert basis plus per-expert Views jointly specializes expert outputs and self-activation without a centralized top-k router, using fewer bytes than independent routing-free experts.
- **T:** Seed 79201; four 8D modes and experts; 256 examples/mode; shared rank-1 output direction; σ=0/.05. Compare independent routing-free experts, hard tie, central top-1, Mirror and native low-rank coordinates. Actual NPZ bytes, routing accuracy, expert load, output nRMSE and MACs; zero optimizer updates.
- **D:** **FAIL** for Mirror-specific value. Balanced self-activation and aligned output quality pass, but Mirror payload is 5.6% larger than independent and 18.8% larger than central top-1; native low-rank exactly aliases. At σ=.05, output nRMSE is .238.
- **C:** Gaussian prototype scores solve routing in this synthetic task, while the shared rank-1 function family is teacher-aligned and ordinary low-rank conditioning.
- **U:** Learned self-activation and adaptive balance, trained nonlinear experts, natural token distributions, load under domain shift, and GPU latency.

## Scope

Synthetic Gaussian modes and linear maps only. No trained Routing-Free MoE, self-activation learning, language loss, or adaptive balancing algorithm is reproduced. Fresh stayed sealed after the exact native low-rank alias.
