# MA-446 status

**FAIL — learned schedule improves mean quality over Adam, but misses the per-world margin and byte gates.**

## H / T

Tested a learned coordinatewise per-step gradient multiplier schedule on a 2D Mirror code against development-tuned SGD and Adam on fresh linear regression tasks.

## D — Fact

At step 4, mean NRMSE: learned 0.4062, Adam 0.4882, SGD 0.5697. Learned/Adam per-world NRMSE: 0.3739/0.4767, 0.3403/0.4758, 0.5042/0.5122; the third world misses the preregistered 0.90x gate. Actual serialized inference payload averaged 101.25B/task for learned vs 91.65B/task for Adam at N=20.

## C — Strongest counter-hypothesis

The schedule may overfit the development task distribution; its small quality gain in the third fresh world does not justify extra policy bytes. Adam remained more consistent.

## U — Unknown

Recurrent learned optimizers, nonlinear models, longer horizons, natural tasks, and whether a shared amortized policy can change the byte frontier are untested.
