# MA-468 status

- Status: FAIL under quality and byte gates; private boundary measured
- Branch: `research/ma-468-polytropon-mirror-skill-views-20261009`
- Base commit: `f2dfb165`
- Initial protocol freeze: `ea565246`
- A1 corrected-control freeze: `2f26575d`
- Valid fresh worlds: 46820–46822 × seeds 0–2
- Initial worlds 46810–46812: exploratory, excluded for incorrect serialized full-bank route map

## H / T / D / C / U

- H: Two physical skill modules with Views can cover four logical skills and held-out compositions at lower bytes than four physical modules.
- T: Four skill vectors, three rotation-aligned and one off-orbit; two-skill routed tasks; no-view, Mirror, Mirror+private, and four-module controls.
- D (Fact): N6 held-out NRMSE: no-view 0.511, Mirror 0.375, Mirror+private 0.00101, full bank 8.98e-8. Payloads: 2,209B, 2,461B, 2,713B, 2,209B. View search cost 184,832 MAC.
- D (Interpretation): Views improve over hard tying, but miss full-bank quality and the byte gate. A private residual repairs the off-orbit skill at a larger payload.
- C: Deliberately aligned synthetic skills favor Views; the private boundary may move on natural skill distributions.
- U: Learned allocation, natural tasks, and larger skill functions remain untested.

## Next action

Commit evidence, update registry/ledger, push, then continue to MA-469.

## Blockers

None.
