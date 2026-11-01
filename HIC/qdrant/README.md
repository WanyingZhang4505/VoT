# DeepSeek Test 1 Qdrant snapshot

`DeepSeek_Test_1_718_20261101.snapshot` is a lossless Qdrant collection snapshot downloaded from the live `DeepSeek Test 1` collection on 2026-11-01.

- 718 points; vectors are 1024-dimensional Cosine.
- Source Qdrant version: 1.19.2.
- Snapshot size: 15,307,264 bytes.
- SHA256: `CF204937015ADDAF7A0BECE6EC92063ECF1F66A4B033C6FBBDCE902D8ADC99F1`.
- The snapshot includes 15 validation-derived experience points appended after the original 15-case evaluation. Treat this as a development memory bank, not independent blind-test evidence.

Restore it to a fresh collection using Qdrant's uploaded-snapshot endpoint; see the project README for a PowerShell-compatible command. Qdrant documents that uploaded recovery can replace data in an existing collection, so inspect the target or use a new collection name first. Snapshot restores require the same Qdrant minor version or the next minor version.
