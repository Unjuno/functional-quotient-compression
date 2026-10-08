# MA-309 — MIMO subnetworks with Mirror member views

Status: protocol frozen before development. Dedicated branch: `research/ma-309-mimo-mirror-member-views-20261008`.

## H — hypothesis

Four member-specific Givens views on a single shared MIMO trunk and classifier may preserve each synthetic task's accuracy and member prediction diversity while using fewer actual bytes than ordinary independent MIMO heads. A scalar output gate is the cheapest non-Mirror control.

## Mirror insertion

PA42's MIMO method obtains multiple functionally distinct subnetworks inside one physical network and evaluates their member outputs together. This experiment places one learned angle per member between the shared hidden trunk and shared output head. It compares standard MIMO member heads, hard tying, scalar output gates, Mirror views and independent MLPs.

## Frozen protocol

See `PROTOCOL.json`. Four binary classification rules on 2D Gaussian inputs; 4,096 train, 1,024 validation and 2,048 test examples/member; batch 64/member and 500 Adam updates. Development seeds 30901/30902; fresh seeds 30911/30912/30913. Fresh worlds stay sealed until source/settings are committed.

## Development observations

On both development seeds, standard MIMO heads reached 0.976–0.984 macro accuracy, while Mirror reached 0.843. Mirror payload was 4,652B vs 6,256B (25.6% smaller); pairwise disagreement was 0.433–0.473 vs 0.500–0.506. Because accuracy missed the frozen 2-point margin, settings remain unchanged and fresh evaluation is a confirmatory negative screen.
