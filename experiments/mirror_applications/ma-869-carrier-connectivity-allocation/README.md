# MA-869 — carrier-connectivity-aware code allocation

This PA233 screen tests whether an observation code chooses a reachable sensor when the best-reliability carrier is disconnected. It charges topology/reliability metadata, policy code, relayed message bytes, and hop latency. The protocol is in PROTOCOL.json.

**Mirror insertion:** this experiment adds the Mirror address `m` at the query-to-carrier selection interface so that a logical observation query can select a carrier without duplicating private sensor models.
