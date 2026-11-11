# MA-866 results

## Fact

- A synthetic Gaussian latent vector had eight dimensions; 32 independent private agents each observed one coordinate, with four noise levels per coordinate (σ=.25,.5,.75,1). Queries requested one coordinate at a time. Only agent ID, typed coordinate ID, and one float32 observation were sent.
- Across 4,096 heldout latent draws and fresh seeds 503/521/547, mean normalized MSE was .0417–.0422 for centralized posterior inference, .0581–.0595 for the best one-agent policy, and .3166–.3224 for random one-agent selection.
- For eight logical queries, centralized mode sent 256 messages / 1,536 B; one-agent selection sent 8 messages / 48 B: 32× fewer messages and message bytes. Both use one-hop star delivery. Selected inference used 24 scalar operations vs 768 for central aggregation.
- Actual complete Mirror program bytes were 38 B (24 B schema + 14 B address table). The native deterministic type/reliability rule was 24 B and chose the same agent IDs. The single wire message was exactly 6 B. No agent weights, gradients, or hidden state were serialized or transmitted.

## Interpretation

A compact admission policy can move along a clear communication-quality tradeoff: the best single sensor uses 1/32 the messages while increasing normalized MSE by about .016 versus the centralized posterior. Random selection is much worse but still below the fixed .35 threshold in this reliability schedule.

The Mirror address table adds no functional behavior beyond native routing and costs 14 B more than the direct type/reliability rule. The communication reduction comes from admitting fewer observations, not from Mirror-specific inference capacity.

## H / T / D / C / U

- **H:** A small query-to-agent address code can select useful private observations with large message savings while preserving observation-only information exchange.
- **T:** 8-D linear-Gaussian latent; 32 agents; four sensors per coordinate; 8 coordinate queries; fixed one-message budget; fresh seeds 503/521/547; centralized, best-agent native/Mirror, and random controls. Actual policy/message bytes, scalar updates, and hop count measured.
- **D:** **FAIL** for Mirror-specific attribution; the communication/quality screen itself passes its fixed gates. Mirror and native routing choose identical agents and outputs, while the native rule is 14 B smaller.
- **C:** The policy encodes a deterministic reliability lookup already derivable from public observation type and noise class; no new functional freedom is needed.
- **U:** General graph routing, heterogeneous nonlinear agents, multiple observations per query, privacy leakage analysis, learned admission policies, and neural collective inference remain untested.

This is a synthetic distributed-inference systems result, not a neural-network parameter-compression result.
