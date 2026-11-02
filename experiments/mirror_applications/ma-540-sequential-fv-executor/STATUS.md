# MA-540 status

**FAIL — frozen development screen; fresh worlds sealed.**

- **H:** A support-extracted function vector at each step lets one shared transition execute held-out ordered operator pairs at high exact accuracy while using <=90% of a same-width native code control's inference bytes.
- **T:** Dev worlds 54001 and 54002; eight random GF(2) affine maps over 16 states; 64 support pairs/world; 2,500 updates x batch 64 for learned methods. All 16 states for each of four held-out ordered pairs were evaluated. Fresh 54011–54013 were not accessed.
- **D:** FAIL. FV and native tied executors both reached 100% held-out exact accuracy and valid paths. FV payload was 27,441 B versus 23,865 B native (1.150x; frozen limit 0.90x); FV active proxy 41,984 MAC/packet versus 9,216. One-shot FV accuracy was 9.4%/28.1%, indicating the explicit state-passing interface mattered in this fixture.
- **C:** A learned operator-code bank plus ordinary recurrent state passing explains all successful composition, with lower storage and compute than FV extraction.
- **U:** Natural language, longer sequences, unseen operators, GPU behavior, fresh replication, and the private-state frontier remain unknown.

Initial development payload/MAC accounting omitted operator-order metadata and one transition step. Those runs are retained under `runs/dev_initial_metrics_undercharged/`; identical frozen seeds/settings were rerun with accounting corrected before final judgment. The amendment and exact provenance are recorded in `PROTOCOL.json` and `VERIFICATION.json`.
