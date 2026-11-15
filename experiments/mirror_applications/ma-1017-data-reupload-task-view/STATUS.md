# MA-1017 status

Stage: COMPLETE — FAIL for Mirror-specific value (aligned synthetic statevector/shot screen).
Branch: `research/ma-1017-data-reupload-task-view-20261115`.
Base evidence: MA-1016 branch tip `acce6c207`.
Fresh seeds: locked and unopened.
H: task phase at each re-upload recovers four functions with smaller payload than independent VQCs, beyond ordinary scalar conditioning.
T: two-qubit, three-block schedule; four distinct tasks; 16 inputs/task; 1,024 shots/input; independent, no-view, Mirror and scalar controls.
D: FAIL. Mirror and native scalar controls exactly alias at 379 B; both preserve teacher and save 27.5% vs independent bank. Five tests/replay pass.
C: only aligned synthetic tasks; fixed task phases encode the teacher family.
U: noise, QPU, compilation, natural tasks, learned adaptation, fresh data.
Last verified commit: `84c1c7928caf2cce36b08d2e8f944467b4ce80e7`.
