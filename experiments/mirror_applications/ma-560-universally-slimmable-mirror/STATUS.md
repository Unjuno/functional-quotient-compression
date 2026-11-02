# MA-560 status

- Status: SCREENING (development complete; fresh locked)
- Branch: `research/ma-560-universally-slimmable-mirror-20261102`
- Protocol froze before implementation/development at `1831195c`.
- Development freeze: `DEV_FREEZE.json`; fresh worlds 56011–56013 remain unopened.

## H / T / D / C / U

- **H:** A compact continuous angle code on shared width-elastic hidden weights can preserve width-conditioned functions at unseen widths with a better quality/bytes frontier than prefix-only US-Net and a generic direct coefficient law.
- **T:** Fixed shared 16→32→16 network; train widths 8/16/24/32; held-out widths 4/12/20/28; two development worlds; 512 train and 512 held-out examples per width; no optimizer updates. Fresh worlds are locked.
- **D:** Development quality passes for Mirror (mean/max NRMSE 6.9e-9/1.57e-8) and beats prefix US-Net and the direct coefficient law, but actual payload is 5,396B vs 9,032B independent (59.8%), missing the frozen <=50% gate. Fresh result pending.
- **C:** The teacher was generated from the same affine Givens orbit used by Mirror; the gain may be an aligned reparameterization rather than useful width-view generality. The generic direct coefficient law costs only 14B more but has higher error.
- **U:** Fresh transfer, final status, measured wall-time Pareto, and behavior on trained US-Net/Transformer weights.
