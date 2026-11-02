# MA-565 — LayerDrop depth + Mirror surviving-layer roles

CPU synthetic mechanism screen based on PA110. The environment has no pretrained LayerDrop checkpoint or GPU. A shared four-block nonlinear network is evaluated at four prefix depths, with depths 2 and 3 held out. We compare native prefix inference, a static rank-one output residual, a two-parameter affine depth-angle Mirror code and independent effective maps. The teacher is Givens-aligned, so any positive result is narrow.


## Fresh result

Across three fresh seeds, Mirror achieved mean/max heldout NRMSE 1e-15/3.7e-15, versus 0.0430/0.0900 native and 0.0438/0.0900 rank-one. The independent upper used 12695 B; Mirror used 7529 B (59.3% of independent), missing the preregistered <=50% total-payload gate. **Disposition: FAIL overall, with a narrow aligned synthetic quality signal.**
