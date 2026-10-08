# MA-332 — Mirror permutation-orbit audit

Status: protocol frozen before development. Dedicated branch: `research/ma-332-permutation-orbit-audit-20261008`.

## H — hypothesis

Consistent hidden-unit permutations of a trained ReLU MLP preserve its function. A compact shared checkpoint plus paid permutation addresses can represent many parameter coordinate variants, but those variants are not additional functions.

## Mirror insertion

> **Mirror insertion:** apply a permutation to hidden activations while permuting incoming columns, hidden bias entries, and outgoing rows consistently.

- Native control: a single trained MLP checkpoint.
- Exact coordinate: a paid vector of hidden-unit indices.
- Counterfactual control: permuting incoming weights only must alter the function.

## Prior art

PA37 and PA47 describe function-preserving permutation/monomial symmetries. This is an implementation audit of whether a candidate Mirror address changes function, not an architecture proposal.

## Frozen protocol

See `PROTOCOL.json`. One small trained 8-12-3 ReLU MLP, eight parameter permutations, actual serialized bytes, and held-out output differences. Fresh seeds are sealed.

## Results

Pending.
