# MA-320 — Tucker logical experts with Mirror phase addresses

Status: protocol frozen before development. Dedicated branch: `research/ma-320-tucker-logical-experts-20261008`.

## H — hypothesis

A small phase address per expert over a shared Tucker bank will encode a 48-expert harmonic orbit with at least 10% fewer total bytes than rank-16 free coefficients while preserving every task's held-out function nMSE <=1e-4. Sixteen off-orbit experts should reveal where private parameters become necessary.

## Exact insertion and controls

Each of 64 logical top-1 experts is a 32-to-16 linear map. The shared physical state is one mean matrix and 16 common matrix atoms. The 48 aligned functions use a shared 8-harmonic coefficient generator and one phase per expert; 16 unrelated functions may require full private matrices. Controls include hard tying, direct Tucker coefficient ranks 1/2/4/8/16, Tucker rank-16 with the same validation-selected full-matrix fallback, and independent full weights. Expert ID metadata is charged; routing is oracle-supplied and not claimed.

## Frozen conditions

See `PROTOCOL.json`. Each expert has 96 support, 64 validation and 128 fresh test inputs. Development seeds are 32001/32002; fresh seeds 32011/32012/32013. No optimizer updates. The 1e-4 private fallback threshold and 1024-point phase search are fixed. Actual fixed-timestamp ZIP/NPY bytes include the shared bank, codes, amplitudes, private matrices, IDs and headers.
