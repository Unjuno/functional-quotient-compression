# MA-641 status

- Status: FAIL
- Branch: `research/ma-641-episodic-adapter-bank-mirror-20261104`
- Base commit: `0567d264`
- Protocol frozen before development: yes (`freeze.json`)
- Development complete: yes (seeds 64101, 64102)
- Fresh/audit opened: no; frozen failure gate reached

## H / T / D / C / U

**H:** A shared rank-4 output basis plus per-episode coordinates compresses correlated episodic sparse adapters while retaining recall; increased private variation reveals when storage savings break down.

**T:** 128D synthetic query-to-answer adapter bank; N=16/64/128, private residual RMS=0/.05/.1/.25, two development seeds. Compared independent sparse rank-1 episode adapters, shared rank-4 Mirror basis/codes, and native rank-4 PCA with the exact same router and decoder. Serialized complete payload includes a fixed dense base executor, router schema, episode keys, adapter state, and metadata. Measured full/state bytes, output nRMSE, cosine-threshold recall, cross-key router leakage, operation proxy and wall time; 0 optimizer updates. Fresh seeds 64111–64113 remain sealed.

**D:** FAIL for Mirror-specific claim. At zero private residual, N=64 and N=128 preserve recall and answer output (mean nRMSE <1e-7), and average full payload is 70.4% and 53.7% of independent adapters. N=16 full payload is 92.5%. Across residual RMS .05/.1/.25, mean error increases to about .05–.06/.11–.12/.25–.29; cosine-threshold recall drops to roughly .94–.98/.73–.94/.05–.13 depending on N. Rank-4 decode uses 4x the operation proxy of the independent adapter. Mirror basis and episode codes are elementwise identical to native PCA; both recover exactly the same function. Serialized container sizes differ by only -6 to +3 bytes between the two identical states across rows. The Mirror-specific gate failed, so fresh was not opened.

**C:** This is ordinary rank-4 PCA compression of correlated adapter outputs; all storage savings are explained by generic low-rank coding. The fixed dense base dominates smaller banks, while private episode directions hurt recall without changing rank-4 storage materially.

**U:** The synthetic exact-key router does not test competitive retrieval under noisy or semantic keys; this is not a PA152 or language-model reproduction. The preregistered metric said nearest-answer exact recall, but implementation reported cosine >=.99 recall instead. That metric deviation is disclosed; nRMSE, actual bytes, and exact Mirror/native alias independently establish failure of the Mirror-specific gate. Fresh performance is not established.

## Fact / Interpretation / Hypothesis

- **Fact:** 72 development rows (2 seeds x 3 bank sizes x 4 residual levels x 3 methods); all actual `.npz` sizes and output metrics replay. Three tests pass. Protocol hash matches `freeze.json`. Fresh seeds untouched.
- **Interpretation:** A generic shared low-rank basis compresses highly correlated episodic outputs as bank size grows, but Mirror provides no additional function or storage benefit over native PCA and costs more per-query multiply-adds.
- **Hypothesis:** Larger real adapter banks may yield stronger generic basis amortization; meaningful Mirror value would require a distinct measured advantage over matched PCA/low-rank controls and recall under realistic retrieval interference.
