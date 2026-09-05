param(
    [string]$RepoRoot = "C:\Users\samsung\KirkyGPT"
)

$ErrorActionPreference = "Stop"

$PatchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path (Join-Path $RepoRoot ".git"))) {
    throw "RepoRoot does not look like the KirkGPT Git repository: $RepoRoot"
}

Write-Host "Applying KirkGPT deterministic rule-engine patch..."

$copyPaths = @(
    "backend\src\llm_followups\rules\__init__.py",
    "backend\src\llm_followups\rules\models.py",
    "backend\src\llm_followups\rules\definitions.py",
    "backend\src\llm_followups\rules\engine.py",
    "backend\src\llm_followups\rules\runtime.py",
    "backend\tests\unit\test_rule_engine.py",
    "backend\tests\unit\test_rule_runtime.py"
)

foreach ($relative in $copyPaths) {
    $source = Join-Path $PatchRoot $relative
    $target = Join-Path $RepoRoot $relative
    $targetDir = Split-Path -Parent $target

    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    Copy-Item -Force $source $target
    Write-Host "  wrote $relative"
}

$mainPath = Join-Path $RepoRoot "backend\src\llm_followups\server\main.py"
$mainText = Get-Content $mainPath -Raw

if ($mainText -notmatch "rules_only_enabled") {
    $pattern = '(?s)    if runtime is None:\r?\n.*?    else:\r?\n        base_runtime = runtime\r?\n\r?\n    # ==================================\r?\n    # Redis generation cache'

    $replacement = @'
    if runtime is None:
        # Rule-only mode preserves the existing runtime contract, so
        # FastAPI, SQLite and Redis do not need a separate code path.
        from ..rules.runtime import (
            RuleRuntime,
            rules_only_enabled,
        )

        if rules_only_enabled():
            base_runtime = RuleRuntime(
                settings
            )
        elif settings.inference_base_url:
            base_runtime = (
                RemoteInferenceRuntime(
                    settings
                )
            )
        else:
            # Existing model-backed path remains available for later.
            # Set RULES_ONLY=false when you are ready to use it.
            from .llm_runtime import (
                LLMRuntime,
            )

            base_runtime = LLMRuntime(
                settings
            )
    else:
        base_runtime = runtime

    # ==================================
    # Redis generation cache
'@

    $updated = [regex]::Replace(
        $mainText,
        $pattern,
        $replacement,
        1
    )

    if ($updated -eq $mainText) {
        throw @"
Could not find the expected runtime-selection block in:
$mainPath

No main.py changes were made. Your local main.py differs from the version this
patch was designed for. Send that file back and it can be patched safely
without overwriting unrelated local work.
"@
    }

    Copy-Item $mainPath "$mainPath.rule-engine.bak" -Force
    Set-Content -Path $mainPath -Value $updated -Encoding utf8
    Write-Host "  patched backend\src\llm_followups\server\main.py"
    Write-Host "  backup: backend\src\llm_followups\server\main.py.rule-engine.bak"
}
else {
    Write-Host "  main.py already contains rule runtime wiring"
}

function Ensure-RuleSetting {
    param(
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        New-Item -ItemType File -Path $Path -Force | Out-Null
    }

    $content = Get-Content $Path -Raw

    if ($content -notmatch '(?m)^RULES_ONLY=') {
        Add-Content -Path $Path -Value "RULES_ONLY=true"
        Write-Host "  added RULES_ONLY=true to $(Split-Path -Leaf $Path)"
    }
    else {
        Write-Host "  $(Split-Path -Leaf $Path) already has RULES_ONLY"
    }
}

Ensure-RuleSetting (Join-Path $RepoRoot ".env")
Ensure-RuleSetting (Join-Path $RepoRoot ".env.example")

Write-Host ""
Write-Host "Patch applied."
Write-Host ""
Write-Host "Verify with:"
Write-Host "  cd `"$RepoRoot\backend`""
Write-Host "  pytest tests/unit/test_rule_engine.py tests/unit/test_rule_runtime.py -v"
Write-Host "  pytest -v"
Write-Host ""
Write-Host "RULES_ONLY=true means the default application runtime does not load or call a model."
