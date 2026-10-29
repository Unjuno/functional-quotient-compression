# MA-1181 NoPE x RoPE x Mirror placement (2026-10-30)

**Mirror insertion:** Learn two pass-specific Householder Q views or two residual-stream boundary views on one shared two-pass causal Transformer, without duplicating the block weights. Test their marginal effects with RoPE versus NoPE.

H: NoPE may remove an adverse interaction between RoPE and learned Mirror rotation on Q, improving the **incremental** value over matched FiLM. The opposite is possible, and shared Q/K Mirror is analytically invisible under NoPE.  Two tasks distinguish content-based from position-based benefits.

Controls: shared no code, FiLM query modulation with matching code count, Q-only Mirror, and residual-state Mirror. Two positions x two tasks x three seeds = 48 models. Train before looking at fresh; preserve all paired unfavorable observations. Never assert NoPE helps MLPS without RoPE.

Check the fixed PROTOCOL.json and source. Status is PRE_REGISTERED_BEFORE_TRAINING. H gates are screening only; independent audit would need new tasks/seeds and larger budgets.
