# MA-938 — Fisher-guided private versus Mirror allocation

## H — hypothesis

After NerVast chooses shared weights with its native Fisher procedure, some remaining chunk-private parameters may be replaced by small structured codes, with sparse private residuals retained only for sensitive parameters.

## T — work completed

Reviewed PA268 and its direct NerVast allocation design. Resource inspection on this worker found no NerVast implementation, fitted weights, Fisher sensitivity/masks or matching video frames; this environment has no CUDA/NVIDIA GPU. No proxy mask was substituted because that would change the experiment’s physical-sharing baseline.

## D — UNTESTED

The required native masks/checkpoints are absent. The experiment cannot measure actual bitrate, RD loss or private-state frontier without them.

## C — strongest counter-hypothesis

A small synthetic Fisher-selected mask could test a generic parameter-allocation mechanic, but it could not establish which NerVast private parameters are replaceable or compare against its native mask.

## U — unconfirmed

All registered quality, bitstream, compute, FPS, and private-residual frontier outcomes.
