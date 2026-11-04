# MA-641 — Episodic adapter bank Mirror compression

Status: SCREENING. This mechanism screen tests whether correlated episode-specific sparse adapters can share an output basis and per-episode coordinates.

## H

A rank-4 shared output basis with episode Mirror codes compresses correlated episodic updates while retaining exact episode recall. Increasing private residual should reveal when independent directions are needed.

## Mirror insertion

The episode update is an output vector applied at a fixed episode query key. A shared basis `B` replaces repeated output vectors `Y`; each episode stores code `c_i`, and decodes `B c_i`. The episode key/router is identical and charged for every method.

## Controls

Independent sparse rank-1 adapters store each key ID and full output vector. Native rank-4 PCA uses exactly the same basis decoder and codes as the Mirror candidate. The fixed base executor and router metadata are serialized in every full inference payload.

## Protocol

See `PROTOCOL.json`. Development seeds are 64101 and 64102. Fresh seeds 64111–64113 remain sealed until all frozen gates pass. The task is synthetic and does not establish language-model editing or PA152 performance.
