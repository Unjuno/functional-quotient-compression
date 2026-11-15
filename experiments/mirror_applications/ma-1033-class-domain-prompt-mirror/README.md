# MA-1033 — Class × domain prompt composition

Status: SCREENING. This is a small frozen-CLIP CPU screen of whether a class coordinate and a domain coordinate can compose into useful prompt views for held-out Cartesian class-domain cells.

The task uses CIFAR-10 with four deterministic image transforms and `openai/clip-vit-base-patch32`. A shallow two-token context is shared across classes and domains. The Mirror arm applies a class factor times a domain factor as a Givens angle to the paired text context tokens. The vision prompt is the shared learned projection of the same physical context. A matched additive `a_class + b_domain` angle control, a shared coupled prompt, zero-shot CLIP and an independent class-domain context bank are included.

This is not an official CoOp, CoCoOp or MaPLe reproduction. MA-1024 through MA-1026 already test those neighboring prompt families; this experiment isolates Cartesian class-domain factor composition. The primary evaluation omits all training images from one predeclared domain per class, while keeping each class and domain represented elsewhere. Test pixels are first accessed after all prompts are trained and serialized.

See `PROTOCOL.json` for frozen gates and scope, `RESULTS.md` for H/T/D/C/U, and `VERIFICATION.json` for replay checks.
