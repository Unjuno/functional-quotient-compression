# MA-618 — add functions to a frozen interpreter using codes

Status: **FAIL** on the frozen development byte and compute gates; fresh remained sealed. Prior art PA128, Neural Interpreters.

## Fact

Two development worlds each used 16 support examples for 32 functions (24 on-orbit and 8 with a private fifth-harmonic residual). The final amended Mirror code achieved mean nMSE 1.88e-6 and 1.33e-6, respectively. Direct coefficient codes were near numerical zero (4.30e-14 / 4.55e-14); the full independent Fourier codes were 1.51e-11 / 2.39e-13. Actual bank payloads were 2,777 B Mirror, 2,841 B direct coefficients, and 2,597 B full coefficients. Mirror fit proxy was 13,271,040 operations versus 49,152 direct and 61,440 full. Measured fit+query wall was 12.64/13.22 ms Mirror, 7.37/6.92 ms direct and 2.92/2.41 ms full. Same-seed support and payload replays are exact for eight final records. No optimizer updates or fresh worlds were used.

## Interpretation

The 64 B (2.3%) saving versus direct coefficients misses the required 20% margin. The complete Fourier code is both smaller and more accurate after paying serializer metadata. Phase search adds roughly 270× the predeclared operation proxy and is slower in measured fitting. Thus the frozen shared interpreter can represent the known analytic family, but this screen does not establish a useful Mirror-specific storage/compute point.

## H / T / D / C / U

- **H:** new functions can be added from support examples using a compact signature+phase code and sparse private residual without interpreter weight updates.
- **T:** two deterministic development seeds, 32 functions each, 16 support/512 query points; Mirror phase search + residual, direct coefficient fit + same residual, full Fourier code and no-adaptation control. Amendment 1 corrected omitted-variable bias in joint residual fitting; Amendment 2 added fit/query wall-clock instrumentation before any fresh access.
- **D:** **FAIL** for the preregistered byte/compute gate; fresh sealed.
- **C:** the gain is a fixed analytic Fourier basis; native full coefficients are already compact and avoid phase search.
- **U:** learned interpreter, natural functions/tasks, variable support, larger banks, fresh replication and optimized decoder kernels.

This is a deterministic support-adaptation screen, not learned Neural Interpreter capacity evidence.
