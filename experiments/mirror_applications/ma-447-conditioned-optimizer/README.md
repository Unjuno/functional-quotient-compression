# MA-447 — Mirror-conditioned learned optimizer

## H — Hypothesis

A compact domain Mirror code can condition one shared learned update policy to provide distinct adaptation behavior across two task families, improving over a single unconditioned schedule while using fewer policy bytes than two independent schedules.

## T — Planned test

Two 2D linear-regression support-design families (isotropic and axis-anisotropic), with held-out task IDs and three fresh worlds. Compare tuned Adam, one learned schedule, two independent learned schedules, and one shared schedule conditioned by a paid domain code.

## D — Pending

Protocol frozen; implementation and results pending.

## C — Strongest counter-hypothesis

A domain-specific scalar learning rate may capture the full benefit without a learned conditioned optimizer.

## U — Unknown

Whether conditioning helps across fresh domains and whether the policy-code byte savings survive serialization.
