# Robot action additive-Givens View diagnostic (2026-11-11)

## Scope

This diagnostic covers only the synthetic additive-angle action-policy path tested by MA-806 and MA-807. It does not establish failure of invertible policies, one-step flow matching, multi-robot transfer, or private skill residuals generally.

## Facts

- MA-806: shared 8D one-step policy plus two Givens coordinates reached success 1.0 and used 1,346 B, 39.8% fewer bytes than independent pair maps. Its payload/output exactly aliased native Givens composition and used 18.75% more active MAC. Fresh remained sealed.
- MA-807: a 3×3 embodiment-task grid with five checkerboard pairs held out achieved heldout action nRMSE 8.27e-8 and success 1.0. Complete payload was 1,534 B versus 1,867 B independent (17.8% savings), below the preregistered 25% gate. It exactly matched the native additive Givens payload/actions and used 18.75% more active MAC. Fresh remained sealed.
- MA-806 and MA-807 are two consecutive scoped FAILs sharing the same structural explanation: direct additive Givens composition is a simpler native control and no distinct Mirror mechanism was isolated.

## Interpretation

The tested low-dimensional action variation is already represented by ordinary group-coordinate composition. In this construction the extra layer/action coordinate arithmetic also raises the active compute proxy. MA-807's embodiment/task split does not improve the full storage frontier after IDs, angles, planes, and metadata are charged.

## Hypothesis for follow-up

A Mirror-distinct robot policy experiment should use a different mechanism, such as a nonlinear observation-conditioned orbit, a measurable private-residual boundary on genuinely heterogeneous skills, or a learned task-conditioned flow code that competes with native INN Adapter/hypernetwork baselines. It should not reuse additive Givens angles as the only differentiator.

## Queue decision

Pause direct additive-Givens robot action-code variants MA-806/807 and defer P1 MA-808's direct observation-context generated action View until it has a mechanism and control that do not collapse to standard additive conditioning. Continue to MA-809, whose sparse private skill residual tests the distinct shared-versus-private capacity boundary. No robot or simulator performance is claimed.
