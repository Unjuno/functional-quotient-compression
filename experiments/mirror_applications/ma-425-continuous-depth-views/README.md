# MA-425 — continuous depth/time views

## H — falsifiable hypothesis

A shared vector field whose basis is rotated continuously by depth time will extrapolate beyond observed depths better per byte than a fixed shared field, a simple linear time-conditioned matrix, and discrete depth-specific matrices.

## T — protocol

Fit derivatives from a stable 2D linear teacher with a smoothly rotating vector field over t∈[0,1]. Evaluate RK4 trajectories both within range and extrapolated to t=2. Compare constant shared field, continuous Givens Mirror view, linear time-embedding matrix A+tB, and four discretized depth matrices held at the last matrix beyond the training interval. Count actual state bytes and identical solver NFE.

This is a controlled linear dynamics screen, not Transformer depth evidence.
