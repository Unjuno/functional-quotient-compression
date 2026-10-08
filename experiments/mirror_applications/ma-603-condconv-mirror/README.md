# MA-603 — Input-conditioned Givens views versus CondConv

Status: SCREENING  
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`  
Prior art: PA121 (CondConv), PA122 (Dynamic Filter Networks)

## H — falsifiable hypothesis

For an input-conditioned teacher whose linear map moves along a known Givens orbit, one shared matrix plus a learned scalar Mirror angle can recover the continuous family with useful quality and a better payload or CPU latency point than finite four-basis CondConv. Full dynamic matrix generation is an upper reference; FiLM and a static map test simpler controls.

## T — frozen protocol

Eight-dimensional standard-normal inputs contain a two-dimensional condition slice. The teacher applies a smooth scalar angle derived from that slice to four paired output Givens planes around a fixed world-specific matrix. Two world seeds are development-only; three fresh world seeds and three model seeds form the audit. Five methods are compared: Givens Mirror, CondConv4, FiLM8, a full dynamic matrix hypernetwork, and a static linear map. See `PROTOCOL.json` for the frozen architecture, optimizer, seed split, gates and metrics. Draw29 replay is in `source/draw29_exclusions.json`.

This is a controlled mechanism test, not a convolution/Transformer benchmark. The teacher is intentionally aligned to the Mirror transform.

## D — decision

Pending frozen development and audit.

## C — strongest counter-hypothesis

CondConv's unconstrained basis mixing may represent the orbit with equal or better quality and a simpler/faster inference path; low-dimensional smooth conditioning can also be captured by ordinary FiLM.

## U — boundaries

No claim about natural dynamic kernels, language modeling, large-scale quality, GPU runtime, or Mirror-specific novelty follows from a synthetic aligned teacher.
