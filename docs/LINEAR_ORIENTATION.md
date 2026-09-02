# Linear Weight Orientation

The historical D57 example used a mathematical row-vector convention where a
linear map is written as `x @ W`. PyTorch `Linear.weight`, however, is stored as
`[out_features, in_features]` and the forward map is equivalent to
`x @ weight.T`.

These are now separate concepts.

## Manifest fields

For PyTorch checkpoints, use:

```json
{
  "checkpoint_weight_orientation": "pytorch_linear_weight_out_in",
  "canonical_operator_orientation": "row_vector_x_times_W"
}
```

Older research manifests that already store mathematical matrices may continue
to use the legacy `orientation: row_vector_x_times_W` field.

## Validation

Attention projection shapes are checked in the declared **checkpoint**
orientation. This matters most for non-square K/V projections under GQA/MQA,
where a silent transpose error is otherwise easy to miss.

## Analysis conversion

`canonical_row_matrix` converts a raw checkpoint weight into the row-vector
operator used by FQC structural diagnostics. For PyTorch weights this is a
transpose; for already-canonical row-vector matrices it is the identity.

## Evidence boundary

Changing orientation is an exact representation convention, not a source of
compression. Any derived QK/VO primitive must record which orientation was used
and should pass the module replay witness before it is treated as valid analysis
input.
