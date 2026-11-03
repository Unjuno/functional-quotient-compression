# Amendment 1 — excluded cache API/serialization smoke

Two nonregistered end-to-end smokes (seeds 1 and 2) checked Pythia loading, cache role indexing, quantize/dequantize and next-token evaluation. The first attempt found that Transformers 4.46 requires converting legacy tuple caches to `DynamicCache`; the runner now uses that wrapper. A serialization audit also added the cache shape to every payload's metadata. Both completed smokes are excluded from registered statistics. No hypothesis, data split, NLL/byte gate, candidate family or registered seed changed afterward.
