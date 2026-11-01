# Agent Test 8 reproducibility bundle

This commit adds a portable project archive for the local Agent Test 8 run to the VoT `BILL_Agent` branch.

- `Agent_Test_8_Project.zip`: runner sources, dependencies, the full Agent Test 8 run and follow-up artifacts, source inputs, the initial 703-point Qdrant export, and restore instructions.
- `Agent_Test_8_Project_manifest.json`: per-file sizes and SHA256 checksums for the archive contents.
- `Agent_Test_8_Project_SHA256SUMS.txt`: SHA256 checksum of the ZIP archive.

The bundle intentionally excludes API credentials, Python virtual environments, Hugging Face model weights, Qdrant runtime storage, and unrelated local scratch files. The original frozen evaluation uses the 703-point baseline. The separately documented 718-point post-score memory bank is derived from the 15 validation cases and must not be described as an independent blind-test library.

To unpack, verify `Agent_Test_8_Project_SHA256SUMS.txt`, then extract the ZIP. Follow `agent_test8_project/README.md` for Qdrant restore and runner workflow. The language-model worker used by the saved run is Codex task-agent; this archive does not claim that an arbitrary clone can replay the same responses without an available Codex worker.

The project was packaged from upstream fork branch `bill` and committed here without changing that branch.
