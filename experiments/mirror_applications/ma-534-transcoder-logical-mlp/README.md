# MA-534 — Role-specific logical MLP views over a shared transcoder bank

**Result: FAIL.** On 2,048 held-out Wikitext-2 train tokens, the four-role Givens Mirror's role-balanced MLP-output relative MSE was 0.955803; plain per-role sparse gating was 0.959720 and diagonal gains were 0.940216. Mirror improved only 0.41% versus plain gating (gate requires >=10%) and was 1.66% worse than the diagonal control. Fresh validation was not opened.

The selected 32-atom bank gave a fast, slightly smaller replacement: Mirror deployment 268,615,943 B vs native model 272,437,465 B, a 1.40% reduction. Bank methods used about 354,816 MAC/token vs native MLP 2,654,208; measured Mirror inference was 0.035s vs 0.123s for 2,048 vectors. The quality loss is large (Mirror MSE 0.956). A full top-128 transcoder was more accurate (0.796) but used 608,492,205 standalone bytes and 16.2x native MLP MACs.

The Givens code was learned (mean absolute angle 0.509 rad), so the negative result does not come from an untrained zero View. Plain sparse features still did not represent the dense MLP well, and the simpler diagonal control did better. See [protocol](PROTOCOL.json), [status](STATUS.md), [results](RESULTS_CORE.csv), [payload manifest](runs/dev/seed_53401/PAYLOAD_MANIFEST.json), and [verification](VERIFICATION.json).
