# MA-1036 status

- Status: FAIL (scoped synthetic CPU screen)
- Branch: `research/ma-1036-rabitq-domain-view-20261115`
- Protocol frozen before development: yes; original freeze commit recorded below
- Development: angles selected on seed 103600 only
- Fresh/audit: seeds 103601–103603 complete
- Results and payload replay: complete
- Registry and global summaries: updated; integrity checker clean

Mirror meets the 25% storage reduction against independent rotation banks (42.6%) and stays within 1 pp Recall@10 (−0.13 pp), but the shared single-index control already matches independent quality. Mirror fails to gain the frozen +2 pp in every domain; no domain passes +2 pp on average. Runtime is NOT ESTABLISHED because repeated short-batch throughput results vary substantially.

See `RESULTS.md`, `POST_AUDIT_AMENDMENT.md`, and `runs/`.
