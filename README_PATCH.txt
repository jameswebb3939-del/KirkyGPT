KirkGPT normalization patch

Canonical naming used:
- Visible/product name: KirkGPT
- Python/C++ identifiers and Prometheus metrics: kirk_gpt
- Native extension: _kirk_gpt_native
- Native C++ namespace/include folder: kirk_gpt_native
- CMake constants: KIRK_GPT_NATIVE_*
- Kubernetes/Docker/DNS machine names: kirk-gpt
- Redis prefix: kirk_gpt
- SQLite filename: kirk_gpt.db
- CamelCase classes/alert names: KirkGPT...

Usage (PowerShell):
1. Extract this ZIP.
2. Open PowerShell in the extracted patch directory.
3. Run:
   .\apply_kirkgpt_normalization.ps1 -RepoRoot "C:\Users\samsung\KirkyGPT"
4. Review `git status`.
5. Rebuild the native extension from a clean build directory.
6. Run your backend/native/Kubernetes tests.

The installer preserves an existing ec_pro.db or KirkGPT.db by renaming it
to kirk_gpt.db when kirk_gpt.db does not already exist.
