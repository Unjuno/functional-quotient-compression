# MA-048 status

- Status: SCREENING
- Branch: `research/ma-048-physical4-logical16-attention-20261007`
- Development complete: yes; selected LR 0.003
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: SCREENING

## Development decision

Mirror uses 10,401B vs 18,405B full MHA and 16,093B rank-1, but at LR 0.003 its MSE is 1.315x MHA and worse than rank-1. Fresh remains justified to measure the preregistered tradeoff. Hash manifest is frozen.
