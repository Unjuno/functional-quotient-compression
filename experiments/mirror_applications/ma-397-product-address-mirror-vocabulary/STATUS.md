# MA-397 status

**FAIL by registered scalar-margin gate; other quality/storage/collision gates pass.**

- Mirror resolved all 4-way product-key collision groups at 1.0 accuracy and matched native Hash Embeddings.
- Payload was 11,874 B versus 20,076 B native hash (59.1%); decoded 4,096 unique vectors.
- Scalar control was already 0.9985/0.9695 accuracy. Mirror margins (0.15/3.05 points) missed required +5 points, though NLL was much lower.
- 10 serialized payloads replayed metrics; 4 tests pass. Fresh 39711–39713 remain unopened.
