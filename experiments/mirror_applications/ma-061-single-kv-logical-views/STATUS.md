# MA-061 status

- Status: FAIL at development screen
- Branch: `research/ma-061-single-kv-logical-views-20261007`
- Development complete: yes; selected LR 0.003
- Fresh/audit opened: no
- Results committed: yes (`53e80de855950fb52062e1dd51c3b5c41395c408`); development selection provenance `90ae87a3c36d4a9cee79dca01881cd060266ab7b`
- Verification committed: yes (`53e80de855950fb52062e1dd51c3b5c41395c408`)
- Registry row updated: yes (tracker commit pending)

## H / T / D / C / U

- **H:** one cached physical KV stream plus per-head Mirror transforms recovers logical roles at lower cache cost.
- **T:** synthetic 16D, four-head attention; five controls; aligned and independent KV teachers; one dev world and two LRs.
- **D:** FAIL: MQA had lower model bytes, identical cache bytes/compute proxy, much better aligned MSE, and faster measured CPU inference.
- **C:** angle optimization could improve with a different schedule, but both tested development rates lost to MQA.
- **U:** fresh generalization, decoding bandwidth, quantized cache, NLL and larger models.

**Fact:** 20/20 dev rows replayed; model/cache bytes exact; tests 3 passed; fresh stayed sealed.
**Interpretation:** Mirror views did not improve the MQA cache frontier in this setup.
**Hypothesis:** future cache views must save physical bytes or memory traffic to beat MQA; untested.
