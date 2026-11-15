# MA-1042 — PGM-index domain residual View

Status: FAIL for Mirror-specific attribution. See RESULTS.md for the scoped mechanism result, all gates, limitations and H/T/D/C/U record.

## Research question

Can a small domain coordinate share one physical piecewise learned index across several logical ordered-key domains while retaining exact predecessor answers and updates?

This screen uses four deliberately affine-related synthetic domains. Its observed 74.97% serialized-payload reduction vs four independent indexes is also reproduced exactly by the ordinary affine-coordinate control. The saving is therefore evidence for generic shared-index/coordinate compression only, not Mirror-specific value.

## Implementation and controls

The experiment uses the official Apache-2.0 PGM-index headers pinned to source commit c6fcf3d34e55eb0061b01e2f49dfcbdb711f1407. The experiment code is in source/run_pgm.cpp; third_party/nanoGPT is not modified. Controls are four independent PGM indexes, one shared index with an ordinary affine query mapping, a no-view negative control, and exact sorted-vector predecessor answers. All serialized keys, update logs, affine codes and configuration are included in actual-byte accounting.

The pinned DynamicPGMIndex is used to apply and check insert/delete operations. It has no predecessor-decrement iterator; after updates the final keys are used to rebuild an official static PGMIndex for predecessor queries. This constraint and both costs are disclosed in RESULTS.md.

## Reproduction

Compile with g++17 using the pinned source include directory. From the repository root:

    g++ -std=c++17 -O3 -DNDEBUG -I /path/to/PGM-index/include experiments/mirror_applications/ma-1042-pgm-domain-residual-view/source/run_pgm.cpp -o run_pgm
    ./run_pgm 1050 10000 /tmp/ma1042-audit 8 1

The confirmatory runs and payload files are under runs/audit. The payload reconstruction and integrity verifier is:

    python experiments/mirror_applications/ma-1042-pgm-domain-residual-view/tests/verify_payload.py -v

The frozen development choice and amendment history are in runs/DEV_FREEZE.json and PROTOCOL.json. Superseded audit artifacts remain under runs/exploratory_invalidated and are not confirmatory evidence.
