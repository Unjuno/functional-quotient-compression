# MA-710 — Learn-many LDR atoms then Mirror-compose and prune

Status: **FAIL at development; no valid audit for amended method**  
Evidence lane: MECHANISM/STORAGE/COMPUTE  
Base commit: `54992634`

## Hypothesis

H: Fitting task codes over an overcomplete learned Toeplitz/LDR bank, then physically pruning atoms unused by those codes, can retain held-out function quality with lower total payload than independent Toeplitz operators; post-fit selection may outperform equal-size random pruning and small-from-start.

## Mirror insertion

> **Mirror insertion:** logical task maps are sparse signed combinations of shared learned LDR atoms; unused physical atoms are removed after support fitting and indices remapped.

Controls include independent dense and Toeplitz fits; full atom bank; ordinary sparse coding over the exact same bank; post-fit pruning; equal-size random prune; and a dictionary trained small from the start. The native sparse coding control exactly aliases the Mirror code representation. PA174 motivates learned LDR transforms; PA78 motivates sparse conditioned refinement.

## Protocol amendments and provenance

The first development implementation used a fixed random bank and accidentally executed its full script, exposing audit IDs 71011–71013. Its code, results, and payloads are preserved under `source/attempts/fixed_random_bank_not_learned/`, but are invalid for the learned-bank hypothesis. The IDs are contaminated and cannot validate the amended method.

Before the amended method was evaluated, the implementation learned a 32-atom dictionary from support-fitted task generators. Alpha was compared on development seed 71001; alpha=.1 was selected as the best of .00000001/.1/1.0 for post-fit-pruned quality, although it still missed the frozen gates. All changes and the contamination are documented in [AMENDMENT.md](AMENDMENT.md).

## Results

**FACT:** Development seed 71001, D=32, 24 maps, 32 learned atoms, two sparse codes per map, 128 support and held-out vectors. At private fraction ρ=0, post-fit pruning retained 28 atoms, serialized 8,376B, and mean held-out NRMSE .141. Independent Toeplitz fit used 6,574B and NRMSE <2.5e-8. Equal-size random prune had the same 8,376B and NRMSE .0232; small-from-start used the same bytes and NRMSE .144. The full 32-atom bank with dense coefficients achieved NRMSE <2e-8 at 11,922B. Independent dense was 98,826B. At ρ=.05 and .2, post-fit NRMSE was .213 and .094; independent Toeplitz remained numerical zero. Mirror and native sparse-coding payload hashes matched exactly. Alpha sweep: 1e-8 retained 10 atoms with .504 NRMSE; .1 retained 28 with .141; 1.0 retained 14 with .812. No valid audit for the amended method was run.

**INTERPRETATION:** This setup did not establish an atom-pruning storage/quality frontier. The best development setting retained most atoms, missed the quality threshold, exceeded independent Toeplitz bytes, and lost to random pruning at equal bytes. Dense coefficients restored quality only at a payload larger than independent Toeplitz. Native sparse coding is byte/function-identical, so Mirror-specific value is also falsified in this screen.

**HYPOTHESIS:** More training tasks, a different dictionary objective, or a private residual frontier might change the result; this candidate’s frozen development gates were missed.

**BOUNDARY:** Synthetic sparse combinations of Toeplitz generators, with task families generated from six hidden atoms. No learned neural network, natural-task quality, model capacity, or GPU runtime result. Audit IDs from the invalid fixed-bank attempt are contaminated; no fresh evidence is claimed.

## H/T/D/C/U

- **H:** Learn many LDR atoms, encode tasks sparsely, then prune unused physical atoms while preserving held-out function quality.
- **T:** Amended DictionaryLearning/OMP on support-derived task generators; development seed 71001; ρ=0/.05/.2; dense, independent Toeplitz, full-bank dense code, native sparse code, post-fit prune, random prune, and small-from-start controls.
- **D:** **FAIL at development.** Quality and storage gates both miss; equal-size random pruning is better; native sparse coding exactly aliases.
- **C:** Ordinary dictionary learning, sparse coding, and model pruning explain the mechanism; 24 task samples may be too few to identify a useful shared basis.
- **U:** Whether more training tasks or a different learned LDR objective changes the frontier. The amended method has no valid audit and no natural-task/runtime evidence.
