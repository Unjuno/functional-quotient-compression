# MA-255 — Mirror context superposition for task models

Status: **FAIL at registered development gates**. Fresh worlds 25511–25513 remain sealed.

## H

A one-angle-per-task learned Mirror coordinate should retrieve four task classifiers from one shared parameter vector with at least +0.03 held-out accuracy over fixed-sign PSP, remain within 0.05 mean accuracy of independent models, and use no more actual serialized bytes than PSP.

## T

Protocol was frozen before development (`PROTOCOL.json`, base `cbd8140`). Each world contains four independently generated Gaussian linear binary classification tasks. Four separate linear classifiers were trained for 500 Adam updates each, then compressed into one shared 32-vector using fixed random-sign PSP, fixed random phase, learned phase, or learned dense diagonal codes. A jointly trained tied classifier and the independent models are controls. Development seeds: 25501, 25502. Actual FP16 ZIP/NPY payload bytes include shared weights, code arrays, metadata and headers.

## D

**FAIL.** Learned Mirror mean accuracy was 0.6991 and 0.6954; PSP was 0.6866 and 0.6689. Gains of +0.0125 and +0.0264 missed +0.03. Mirror was about 0.28 below independent accuracy (0.9789 and 0.9757), missing the independent-quality gate. Mirror payload was 830 B versus PSP 822 B (1.010x), also missing the byte gate. Fresh was not opened.

The dense learned diagonal control achieved 0.8008 and 0.8586 mean accuracy at 1,068 B, showing a larger code can recover more task function in this setup. The tied model reached 0.6644 and 0.6835. Independent payload was 754 B; for this very small model, archive metadata and headers erase the apparent raw tensor saving.

## C

The strongest counter-hypothesis is that one global phase per task is too constrained to suppress cross-task interference. A full per-coordinate code is much more expressive, and fixed-sign PSP already provides a richer context than one angle; both facts explain the Mirror's small accuracy gain. The tiny model's archive overhead also makes all compressed payloads larger than the independent payload.

## U

This is an oracle compression/retrieval test on synthetic linear classifiers, not joint learning or neural-network evidence. No fresh worlds were evaluated because development gates failed. Results show neither useful multiplicity nor a storage advantage at this scale.

## Fact / interpretation / hypothesis

- **Fact:** Both development seeds fail accuracy and actual-byte gates. Twelve payloads were reloaded; maximum replay accuracy delta after FP16 serialization was 0.000184. Three tests pass.
- **Interpretation:** The tested scalar-angle Mirror context does not improve the PSP quality/storage frontier; dense task codes improve quality but cost more bytes.
- **Hypothesis:** A structured multidimensional view code might retain a storage advantage while reducing PSP interference, but MA-255 did not test that family.
