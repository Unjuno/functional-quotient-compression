# MA-1175: Corrected Mirror supergraph intervention experiment

**Mirror insertion:** The same MA-1174 128-scalar shared weight core is mapped into 32 logical virtual 8→8→8 MLP edges via four learned addresses and fixed Householder-tanh-inverse views; the only structural intervention is temporary or permanent hard removal of candidate directed edges, tested for its *downstream effect on shared-core learning* rather than present-time gate size.

This is a new preregistered replication of an already FAILED result, not a rewrite of MA-1174.

## Corrections
1. Gradient-conflict shortlist uses exact per-edge gradient contribution to **the shared physical core**, accumulated through two recurrence steps, not cross-batch changes in gate gradient sign.
2. Prospective cutting **always competes against not cutting** and requires a real dev improvement of >=0.0005 NMSE.
3. H=6 versus H=40 prospective probes; a separate H=120 dev-only audit measures possible horizon misranking at the *first* intervention but never selects training candidates.
4. Same initialization/minibatches/teacher across policies; from-start final mask, random, low-gate, true-gradient, direct non-Mirror and compute-matched-extra-update controls.
5. New seeds with sealed independent fresh/OOD. Real execution cost and abandoned probe cost are counted; tests audit exact per-edge VJP sum.

The strongest scientific result would be a paired improvement from structural learning over equally trained fixed-structure, from-start-final-structure, non-Mirror and fully compute-accounted controls. A null result stays FAIL.

Run: `PYTHONPATH=source python -m unittest discover -s tests -v`; then `PYTHONPATH=source python source/run.py train`; then `PYTHONPATH=source python source/run.py audit`.
