# MA-407 — demodulated Mirror-MoE expert views

## H — falsifiable hypothesis

On four context-specific functions sharing one FFN, demodulating the context-addressed Mirror effective weights will reduce activation-scale variation and improve held-out accuracy over the same undemodulated Mirror views, while retaining a useful storage frontier against IA3/rank-one logical experts and independent FFNs.

## T — protocol

Synthetic context classification, four known expert addresses, fixed teacher built from a shared MLP and four context Givens views. Compare shared, Mirror without demodulation, Mirror with row-norm demodulation, IA3, rank-one context residual, and independent experts. Two development worlds choose LR; three fresh worlds are locked. Charge actual serialized inference payloads, context codes, and effective-weight construction MACs. Record per-context accuracy and activation RMS spread.

This is a controlled mechanism screen. It does not establish learned routing or language-model gains.
