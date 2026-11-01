# Reproducible Agent Test 8 HIC project (DeepSeek)

The `HIC/` folder contains the complete sanitized source package for the local HIC workflow and the DeepSeek Test 1 evaluation. `Agent_Test_8_Project.zip` is the same package as a single download; `Agent_Test_8_Project_manifest.json` lists each packaged file and SHA256, and `Agent_Test_8_Project_SHA256SUMS.txt` verifies the ZIP.

For a step-by-step clone, Qdrant restore, DeepSeek configuration, and new-case run guide, read [`HIC/DEEPSEEK_HIC_README.md`](HIC/DEEPSEEK_HIC_README.md). It also distinguishes the fixed 15-case runner from the read-only one-case runner.

The project includes the `DeepSeek Test 1` 718-point Qdrant snapshot, plus the clean 703-point frozen evaluation baseline. The current snapshot uses 1024-dimensional Cosine vectors and was created by Qdrant 1.19.2. It includes 15 experiences derived from validation after scoring, so treat the 718-point library as development memory, not an untouched benchmark resource. Snapshot metadata and restore instructions are in `HIC/qdrant/` and the project README.

API keys, virtual environments, Python caches, and downloaded model weights are excluded. Start from `config/api.env.example` and supply your own key locally. Read `HIC/README.md` before running the strict one-pass runner; it records the collection binding, artifact paths, target limitations, and evaluation boundaries. On Windows, extract the ZIP to a short path such as `C:\HIC` because the frozen audit filenames are long.
