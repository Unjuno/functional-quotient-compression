# MA-1016 status

Stage: COMPLETE — FAIL (aligned synthetic statevector/shot screen).
Branch: `research/ma-1016-tensorhyper-vqc-mirror-20261115`.
Base evidence: MA-1014 branch tip `83f8a136c`; main untouched.
Fresh seeds: locked and unopened.
H: one scalar phase per task can preserve TT-generated VQC outputs while reducing actual storage beyond native task cores.
T: 4-qubit, 2-layer ideal statevector; four distinct task functions; all 16 bit inputs; 1,024 shots/input; independent, rank-2 TT, Mirror phase and native scalar-phase controls; fresh sealed.
D: FAIL. Mirror saves 3.4% vs TT, not 10%; is byte/function identical to ordinary scalar-phase control; independent bank is 48 B smaller. Six tests and deterministic replay pass.
C: Four tasks are too few for TT sharing; compact serializer metadata dominates the small models.
U: trained task generalization, noisy device, compiled circuits, official TensorHyper reproduction, fresh seeds and QPU execution.
Last verified commit: `2a2aa74d5ad9492a5d4b297a72f22dfcb295e5f8`.
