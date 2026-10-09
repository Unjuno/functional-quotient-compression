# MA-516 — Function-vector Mirror compression basis

Status: SCREENING. A pinned frozen GPT-2 checkpoint is used because this workspace had no pretrained checkpoint locally. The model is `openai-community/gpt2` at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` (124,439,808 parameters), CPU float32. Only the selected safetensors checkpoint and inference tokenizer/config files are charged; duplicate converted ONNX/TF/TFLite/Flax exports are excluded.

## H / T

**H:** A task vector extracted from support demonstrations can be compressed into a development-fitted shared PCA basis while preserving held-out task accuracy at lower intervention bytes than explicit task vectors.

**T:** Frozen GPT-2 receives synthetic arbitrary object-to-color mappings: four support pairs and four held-out queries per task. The function vector is the layer-6 residual-stream delta between a demo prompt and a fixed neutral prompt. Compare query-only, direct ICL, explicit vectors, and PCA ranks 4/8/16. Dev worlds 51600/51601 fit the basis; fresh worlds 51610-51612 × seeds 0-2 are locked.

**D:** Pending fresh evaluation.

**C:** GPT-2 may fail to learn arbitrary mappings from four demonstrations; if direct ICL and explicit vectors are near chance, compression cannot establish useful function retention.

**U:** All fresh accuracy, target NLL, intervention bytes, and runtime results.
