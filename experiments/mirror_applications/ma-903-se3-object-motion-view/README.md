# MA-903 — canonical object Gaussians with SE3 Mirror motion code

Status: FAIL for Mirror-specific attribution — scoped synthetic mechanism
Base commit: `a2dfbc4b6dd4b6042a16db972c46a6fb770120d` (MA-902 research branch)

## H

Canonical geometry plus compact SE(3) codes should reproduce unseen rigid action compositions with fewer actual bytes than materialized posed point clouds. Mirror-specific value must beat native SE(3) parameters at matched state.

## T

Four synthetic point clouds, six primitive rigid actions, and all ordered action-pair compositions. Development seeds 90301/90307. Amendment 01 corrects the native control to equal the six-transform Mirror payload and replaces the initially evaluated fresh seeds with 90331/90333/90339; initial results remain archived and exploratory. Exact data and gates are frozen in `PROTOCOL.json`.

## Boundary

This tests geometry sharing and valid rigid transform composition only. It does not run PA252's learned dynamics/world model or establish rendering/interaction quality.
