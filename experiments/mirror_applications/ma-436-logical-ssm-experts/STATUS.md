# MA-436 status: PROMISING (synthetic quality/storage frontier; compute cost remains)

## H — falsifiable hypothesis

A shared transition plus per-role Givens Views would recover token-wise role switching with near-independent sequence quality and fewer actual bytes.

## T — executed

Four stable 8D transitions, role switches at every token, sequence length 32. Compared shared, Mirror logical experts, per-role rank-one residual, and independent matrices. 300 AdamW updates × batch 32; 2 development worlds × 3 seeds selected LR 0.01; three fresh worlds × 3 seeds. Fresh sequences used the same world teacher parameters with independent input samples. Same route labels and 32 recurrent steps; measured actual serialized bytes, active MAC proxy, training wall, and batched inference time.

## D — PROMISING, narrowly scoped

Fresh means: shared NRMSE 0.09639 / 2,277B / 128 MAC-token; Mirror 0.00047 / 2,529B / 192 MAC-token; rank-one 0.05411 / 2,909B / 160 MAC-token; independent 0.00074 / 3,045B / 128 MAC-token. Mirror improves quality over independent while using 17.0% fewer actual bytes, so the synthetic quality/storage frontier improves. It uses 50% more recurrent MAC proxy and 1.56× batched inference wall time than independent, missing the preregistered <=1.5× runtime and <=60% byte gates. This is PROMISING for a bounded quality/storage Pareto point, not a deployment win or capacity claim.

## C — strongest counter-hypothesis

The teacher is generated from the same Givens role family, favoring Mirror. The byte saving is only 17%, while role rotations add recurrent operations; the unfused implementation is slower.

## U — unresolved

Learned routing, longer sequences, larger hidden state, real Mamba/S4 quality, fused selective scans, and throughput on accelerator hardware remain untested. Oracle-known role labels and a synthetic teacher limit generalization.

