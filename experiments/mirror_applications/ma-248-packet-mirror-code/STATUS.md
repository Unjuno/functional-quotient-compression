# MA-248 status

- Status: SCREENING
- Branch: `research/ma-248-packet-mirror-code-20261007`
- Base commit: `f91625f2fe1b110593b16605c26fc9c7675c1824`
- Development complete: yes (v3 matched-minibatch screen; LR 0.01 selected)
- Fresh/audit opened: no; seeds 24801–24803 locked
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Development screen

At selected LR 0.01, correlated-mode Mirror reached 100% exact packets on development world 24800 and outperformed the PTP and scalar-gate runs in that single world; untied also reached 100%. Independent-source exact accuracy remained low for PTP, Mirror, and untied. Proceed to the locked fresh worlds for replication and entropy-boundary measurement.

## Blockers

None.

## Decisions

- PA10 uses one random auxiliary per predicted future position; the test retains that as the direct non-Mirror control.
- Correlated and independent binary branch sources distinguish packet entropy 1 bit from 4 bits at P=4.
- Model payload, packet-address bits, and their sum will be separately reported.
