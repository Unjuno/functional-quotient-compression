# MA-969 — Small Mirror state plus private synapse exceptions

Status: NOT ESTABLISHED; invalid implementation screen preserved. See RESULTS.md.

## Hypothesis

A shared recurrent SNN can retain one event domain and adapt to another with compact threshold state and a sparse private synapse mask, using fewer bytes than keeping an independent target SNN.

## Mirror insertion

A shared 8-neuron LIF core is trained on D1 event trials. D2 receives a scalar threshold coordinate along a shared threshold basis and, in a sweep, a top-k sparse correction to the 8×4 synapse matrix. Direct per-neuron thresholds are the native control; a separate D2 model is the upper control.

## Scope

The benchmark uses actual D1/D2 Vernier event streams, but a compact from-scratch SNN and sign labels. It is not the pretrained PointTransformer hyperacuity result, TACOS, or hardware-energy validation.
