# Amendment 1 — safetensors F16 reader

Before any development or fresh evaluation, inspection of the pinned checkpoint showed `embed_out.weight` is stored as F16. The first implementation accepted only F32 and therefore could not read the declared base model. Updated the parser to accept F16 and F32, preserve the exact checkpoint values, and convert to F32 for deterministic arithmetic. The protocol now records this numerical handling. No world was generated or metric inspected before this amendment. Added a F16 parser unit test. The original implementation is represented in git history.
