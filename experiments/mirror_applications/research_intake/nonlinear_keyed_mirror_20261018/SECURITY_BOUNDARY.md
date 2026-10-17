# Keyed Mirror: condition, access control and cryptographic boundaries

**Status: security threat-model proposal only. This is not a deployed secure model or a proof of encryption.**

## Three distinct interpretations of "key"

| Item | Example | What it controls | Is it secret? |
| --- | --- | --- | --- |
| condition `c` | green vs red signal | select active/nonactive logical function | usually **no** |
| Mirror code `m` | Givens/Householder/spline coordinates | mathematical behavior of one shared parent | not inherently |
| cryptographic key `k` | AEAD decryption key | access to encrypted stored `m` and potentially full model | **yes** |

Writing `m` as a bit string or multiplying by secret orthogonal Q is not encryption. Rotation, Householder products, permutations and elementwise nonlinearities must not be treated as a cryptographic primitive without an independent formal cryptanalysis and accepted construction. For proposed deployment, use a standard authenticated cipher (such as AES-GCM or ChaCha20-Poly1305), high-entropy managed keys, unique nonces, integrity tags, controlled key release and auditable failure modes.

## Attackers and explicitly limited guarantees

| Attacker | Can observe | Proposed defense | Not guaranteed |
| --- | --- | --- | --- |
| disk/backup reader without k | ciphertext for m; possibly plaintext theta | authenticated encryption of m | **theta** remains visible if plaintext |
| unauthorized inference caller | public model API and its outputs | authentication / allowlist / rate controls | query-based extraction or model distillation automatically prevented |
| attacker controlling host process | decrypted m and theta may be available in RAM | none without trusted execution and system controls | parameter secrecy under local compromise |
| attacker observing memory or GPU buffers | runtime states and keys | isolated execution / TEE if supported, with explicit threat model | universal immunity to side channels |
| user with wrong key | ciphertext that fails integrity check | reject before executing; no fallback using unverified m | random wrong-key outputs counted as confidentiality |

**For full weight confidentiality**, encrypt the complete model at rest **and** protect key release, decrypted parameter memory and runtime via a separately vetted trusted environment. A hard-to-interpret Mirror transformed weight file alone does not satisfy that objective.

## What to test

- Authentication and integrity: modified tag/nonce/key fails closed, no plaintext fallback; nonce uniqueness and revocation; no secrets in logs.
- Functional performance: unchanged authorized task accuracy/NLL versus plaintext m, measured initialization/decryption and per-token overhead.
- Unauthorized functionality: off-route false activation and permission denial; separate authorized behavior tests from crypto access tests.
- Inference leakage: black-box cloning with controlled query budget and exposed model metadata, ordinary adapter encryption baseline.
- Accounting: ciphertext and tag sizes, nonces, key references, storage overhead, protected-memory footprint; report theta and m independently.

**Excluded claims until separately demonstrated**: model theft prevention, weight secrecy when theta is plaintext, resistance to a compromised host, cryptographic strength inferred from model accuracy, security from "many mirrors" or from unpredictable activation behavior.

## Dependencies and stage gate

MA-1160 ("green light") is a **conditioning task**, not a security guarantee. MA-1161 may proceed in parallel but should use standard encryption from the start. The key-based idea does not require proving a new cryptographic primitive. An application may encrypt ordinary LoRA or codebooks with identical access protection; the only Mirror-specific gain then could be much smaller protected/serialized feature codes or function access modularity, measured objectively.
