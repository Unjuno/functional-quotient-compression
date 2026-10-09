# MA-450 — View vs private residual allocation

## H — Hypothesis

A development-selected controller can keep related tasks in a shared Mirror subspace and allocate private residuals only where validation indicates the shared view is insufficient, retaining near-full quality with lower serialized storage.

## T — Planned test

Mixed in-subspace and out-of-subspace 16D linear tasks. Compare always-Mirror, always-private, a simple validation threshold, a learned validation controller, and oracle allocation. Development task IDs select the rule before fresh worlds; complete serialized payloads include controller, indices, flags, codes, residuals and metadata.

## D — Pending

Protocol frozen; numerical execution pending.

## C — Strongest counter-hypothesis

A simple validation-error threshold may fully explain allocation behavior; the learned controller may add bytes and compute without improving the storage-quality frontier.

## U — Unknown

Whether support validation reliably predicts the value of private residuals on fresh task families and whether the byte savings survive serialization.
