# MA-436 — logical SSM experts with token-wise roles

## H — falsifiable hypothesis

One shared state transition plus a small per-role Mirror View will recover four expert dynamics that switch at every token, with similar sequence quality and fewer actual bytes than independent SSM experts.

## T — protocol

Synthetic sequence regression with known per-token role labels switching among four stable 8D state transitions. Compare shared tied transition, Mirror-conjugated logical experts, per-role rank-one residuals, and independent transition matrices. Measure normalized state trajectory error, actual payload bytes, recurrent MAC proxy, and batched inference wall time.

This tests expert-state sharing under token-wise switching; routing is oracle-known and not learned.

