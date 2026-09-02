# Real-Model Storage Accounting

The 64× denominator uses `N`, the count of unique paid scalar values in the declared baseline. Therefore alias/tied-storage accounting is part of the compression certificate, not a bookkeeping detail.

## Exact alias rule

Two tensors may share one baseline storage count only when the extractor proves that they refer to the same underlying storage interval with compatible scalar interpretation. A manually repeated `storage_group` label is not evidence.

The canonical storage inventory records:

- underlying storage identity;
- scalar offset;
- scalar length;
- dtype / element size;
- exact-alias interval group.

Exact aliases such as a truly tied embedding and LM head are counted once.

## Partial overlaps

D57's simple equal-size `storage_group` convention cannot safely represent arbitrary partial overlapping views. The canonical default is therefore to reject partial overlaps rather than silently classify them as tied tensors.

If partial-overlap models must be supported, the denominator must be computed from explicit storage-range union accounting and the resulting manifest must preserve those ranges.

## Noncontiguous and dtype-reinterpreting views

Noncontiguous views are not representable by one scalar interval and require explicit index/range accounting. Mixed dtype/element-size interpretations of one storage are also rejected by the simple scalar-range model.

These restrictions are conservative by design: an ambiguous storage relationship must not make the 16N baseline artificially small.
