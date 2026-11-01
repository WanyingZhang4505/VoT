# DeepSeek Test 1 Qdrant snapshot

`DeepSeek_Test_1_718_20261101.snapshot` is a lossless Qdrant collection snapshot downloaded from the live `DeepSeek Test 1` collection on 2026-11-01.

- 718 points; vectors are 1024-dimensional Cosine.
- Source Qdrant version: 1.19.2.
- Snapshot size: 15,562,752 bytes.
- SHA256: `d0fd9a858bc75f76504da492b9d72e152c867e8c69a541736842ae3c42c32860`.
- Restored into a fresh collection and verified at 718 points. Every point ID, payload, and vector matched the source; see `restore_verification_20261101.json`.
- The snapshot includes 15 validation-derived experience points appended after the original 15-case evaluation. Treat this as a development memory bank, not independent blind-test evidence.

Restore it to a fresh collection using Qdrant's uploaded-snapshot endpoint; see the [root guide](../../DEEPSEEK_HIC_README.md) for a PowerShell-compatible command. Qdrant documents that uploaded recovery can replace data in an existing collection, so inspect the target or use a new collection name first. Snapshot restores require the same Qdrant minor version or the next minor version.
