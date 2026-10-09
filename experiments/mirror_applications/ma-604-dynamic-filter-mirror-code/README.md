# MA-604 — Dynamic filter Mirror code generator

Status: SCREENING
Evidence lane: MECHANISM / STORAGE / COMPUTE
Base commit: `16ddd3f2` (worker-ready baseline plus MA-602 and MA-603 evidence)
Prior art: PA122, Dynamic Filter Networks

## Hypothesis

H: For an input-conditioned linear operator used as a non-convolutional Transformer-style projection, a short input-generated Givens code applied to one shared operator can retain quality while using materially fewer serialized bytes and lower operator-generation cost than a generic dynamic-filter hypernetwork.

## Mirror insertion

> **Mirror insertion:** this experiment adds a three-angle input-conditioned coordinate `m(x)` to a shared 8×12 operator, applying two output-space and one input-space Givens rotations so that sample-specific logical filters can be realized without generating and storing a full filter generator's output weights.

Native dynamic-filter method: a hypernetwork emits all 96 filter entries for each input. Stronger shared control: three learned operator-basis matrices mixed by input-conditioned linear coefficients. The Mirror has one canonical matrix and a three-output coordinate generator.

## Gates

PASS requires fresh Mirror MSE within 10% of the shared-basis control in both worlds, at least 25% fewer bytes than the generic full-filter generator, and a per-example operation proxy no more than 1.5× shared-basis control. Otherwise FAIL; fresh remains sealed after a development miss. The oracle teacher is privileged and is not a deployable method.

## H / T / D / C / U

**H — hypothesis:** a three-angle dynamic Mirror code can replace a full dynamic-filter generator with similar held-out quality at materially lower actual bytes and operator-generation compute.

**T — execution:** CPU PyTorch 2.14.1; dynamic 8×12 linear operator `y=W(x)x`; teacher mixes four random operators with nonlinear input-conditioned coefficients. Development worlds 60401/60402; 4,096 train and 1,024 held-out inputs, 1,200 AdamW updates. Compared static operator, one-matrix/two-left+one-right Givens Mirror, native three-matrix coefficient basis, generic full-filter MLP, and privileged oracle. Fresh 60411/60412 remained sealed after the frozen quality gate miss.

**D — FAIL:** Mirror uses 2,657 B versus 16,669 B for the generic full-filter network (84.1% fewer) and 228 versus 7,008 MAC proxy. But normalized held-out MSE is 0.1341/0.3393, versus shared-basis 0.01955/0.04321 (6.9×/7.9× worse) and full-filter 0.02228/0.03901. Although it also costs less than the basis control (3,425 B; proxy 456), it fails the quality gate by a wide margin. Fresh stayed sealed.

**C — strongest counter-hypothesis:** this teacher varies over three independent matrix directions, while a single canonical operator under only three plane rotations cannot represent those changes. The native shared basis captures them directly and is the stronger quality/byte compromise.

**U — boundaries:** one synthetic linear projection and fixed-update screen. No Transformer, natural data, or capacity claim. The byte/compute advantage against full filter generation is a quality-distorted point, not a Pareto improvement.

## Facts / interpretation / hypothesis

- **Fact:** all ten saved serialized payloads match recorded size and SHA-256; reload output maximum absolute error was zero; both unit tests passed.
- **Interpretation:** generating only a short code can greatly reduce cost relative to emitting every filter coefficient, but this particular structured transform cannot span the tested function family. Native basis mixing offers much better quality for 29% more bytes and 2× the proxy work.
- **Hypothesis:** dynamic Mirror code families need transform geometry matched to the operator family's variation; few angles on one base are insufficient for general matrix-valued changes.
