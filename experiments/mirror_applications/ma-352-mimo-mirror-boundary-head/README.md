# MA-352 — MIMO boundary-head compression with Mirror

Status: SCREENING. Prior art: PA42 MIMO implicit ensembles.

## H

With a common MIMO trunk evaluated over all member inputs, compressing only aligned member-specific output heads into a shared head plus Mirror phases may preserve member quality/diversity while reducing total trunk+head payload. A generic shared-basis control may erase the Mirror-specific advantage.

## Mirror insertion

> **Mirror insertion:** this experiment adds a member phase `m` to a shared output head after a common hidden trunk, replacing only the per-member head bank while retaining the shared trunk and multi-input/output training.

Compare shared trunk+full MIMO heads, shared trunk+Mirror head views, shared trunk+generic coefficient basis, and a single shared head. Every payload includes the identical trunk, head state, codes and metadata. Evaluate held-out member NLL/ECE/accuracy/disagreement, total bytes, head bytes separately, and one-batch latency including trunk and view reconstruction.

## T

16-input, 64-hidden ReLU trunk, eight binary member tasks, teacher heads on a planted planar rotation orbit. 256 training and 1,024 test examples/member. Train all methods jointly for 1,000 Adam updates using the same multi-member input batch. Development worlds 35221–35222; fresh worlds 35231–35233.

## Gates

**PROMISING:** Mirror NLL within .02 and disagreement within .02 of MIMO; total payload <=.80x MIMO and <=.90x generic basis; single batched inference retained. **FAIL:** total payload gate misses, quality/diversity falls, or simple basis matches within 10% bytes/quality.

## C / U

The random shared trunk is frozen and the teacher heads are intentionally rotation-aligned. This isolates output-head storage but is not a trained deep MIMO reproduction. Natural head diversity and end-to-end training are untested.


## Results

**H:** compressing only output heads may reduce total MIMO bytes while retaining member quality/diversity.

**T:** three fresh worlds, shared frozen 16→64 ReLU trunk, eight member tasks, 256 train and 1,024 test examples/member, 1,000 joint updates. Multi-member inputs were retained. Payload charges identical trunk state in every method.

**D — FAIL for MIMO diversity preservation:** fresh means: MIMO 6,500 B, NLL .57023, disagreement .02687, correlation .857; Mirror 4,744 B, NLL .52571, disagreement .00003, correlation 1.000; generic basis 5,301 B, NLL .52517, disagreement .00217, correlation .99986; single shared 4,708 B, NLL .52572, disagreement 0. Mirror saves 27.0% bytes versus full MIMO, but the 36 B over shared single-head state adds essentially no member distinction. Batched latency: Mirror 167.9 μs vs MIMO 164.8 μs. Lower NLL reflects shared-model regularization under fixed updates, not recovered ensemble capacity.

**C:** tasks are planted on an aligned head orbit and the trunk is frozen random; member tasks are close enough that the shared head is a strong fit. The head code collapses during joint training.

**U:** a diversity-regularized objective might preserve views but would be a new protocol; natural MIMO heads, trained deep trunk and larger-scale throughput remain untested.

### Fact / interpretation / hypothesis

**Fact:** 12 fresh rows across three worlds and four methods; payload hashes/metrics replay exactly; three tests pass.

**Interpretation:** replacing the member-specific head bank with learned Mirror codes saves bytes by removing the diversity the MIMO bank encoded. Output NLL alone would hide this failure.

**Hypothesis:** explicit diversity constraints or private head residuals may be required; quantify their bytes before claiming MIMO head compression.
