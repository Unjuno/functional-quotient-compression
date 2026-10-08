# MA-319 protocol amendment A1

Date: 2026-10-08. Amendment frozen before any valid fresh seed/audit-corpus evaluation.

## Trigger

The initial development implementation incorrectly evaluated the designated final 10% Tiny Shakespeare audit split while running development seeds 31901 and 31902. This was a source-control bug: development should have evaluated only the middle 10% split. Those two runs and their audit-column metrics are preserved under `protocol_variants/contaminated_pre_amendment/`; they are exploratory and are not fresh evidence. Seed 31911 was interrupted during training before any audit evaluation and produced no result artifact.

## Change

The Tiny Shakespeare corpus remains training/development data with the same fixed 80/10 partition and the same tokenizer. The fresh audit is replaced with UTF-8 Project Gutenberg eBook 1342, *Pride and Prejudice* (`https://www.gutenberg.org/cache/epub/1342/pg1342.txt`). The source bytes are SHA-256 recorded at the first fresh execution. Evaluation maps only characters in the frozen Tiny Shakespeare vocabulary after fixed quote/dash normalization; removed-character fraction and retained count are reported. Fresh models remain seeds 31911, 31912 and 31913. No already-viewed Tiny Shakespeare final-split metric is used for selection or claims.

Development runs now call only the middle 10% Tiny Shakespeare split and do not download, read, hash or evaluate the audit corpus. Fresh evaluation opens the new corpus only after the amended source and protocol are committed. The model, optimizer, 500-update budget, compression methods, payload accounting and gates remain unchanged.

## Status

This is an explicitly retained protocol variant, not a retroactive replacement of the original protocol. It exists because the original audit split was exposed by the implementation bug. Both the original protocol and contaminated exploratory outputs remain in version control.
