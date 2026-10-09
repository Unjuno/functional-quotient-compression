# MA-527 — Mirror transform over SAE feature space

Status: SCREENING. Protocol frozen before data extraction. Prior art: PA102 and PA96.

## H — falsifiable hypothesis

A compact Givens transform over sparse SAE coordinates will preserve held-out relation steering quality while keeping feature activity sparse, and will improve the byte/quality frontier over native SAE top-k, global sparse coding, and LoReFT.

## Insertion point

The intervention transforms a sparse feature activation code immediately before the shared tied SAE decoder maps it back to the residual stream. This isolates Mirror coordinates from neuron-space steering. The transform is parameterized by paired Givens angles; all angles, indices, coefficients and metadata are paid.

## Controls and accounting

Controls and fixed split are listed in [PROTOCOL.json](PROTOCOL.json). The pinned model and complete SAE state are shared across methods and charged once. Method-specific code is measured from actual serialized inference files. Dense residual steering is an upper reference; LoReFT rank 8 is the simple representation-space control.

## Data discipline

Fit tasks choose the shared SAE coordinate support. Development tasks choose only the registered angle and intervention scale grids. Fresh task IDs and seeds remain unopened unless both development seeds meet every gate. Fresh results cannot change the frozen configuration.

## Evidence limits

This experiment tests four relation interventions on one model and layer. A positive result would establish only scoped feature-space steering, not arbitrary behavior capacity or routing.
