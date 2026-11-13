# Amendments and execution notes

Amendment 01 froze the disjoint TinyStories row ranges, feature labels, model revisions, controls, seeds and gates before metric evaluation. No metric-driven hyperparameter or gate changes were made.

Two implementation-only corrections were needed before a complete run: evaluation of inference tensors was placed under `torch.no_grad()`, and the probe serialization replay check was wired into the script. Earlier invocations stopped before writing a completed result table; the final frozen script run produced 96 rows and passed the reload replay. They are not included as experimental outcomes.
