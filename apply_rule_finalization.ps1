param(
    [string]$RepoRoot = "C:\Users\samsung\KirkyGPT"
)

$ErrorActionPreference = "Stop"
$PatchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path (Join-Path $RepoRoot ".git"))) {
    throw "RepoRoot is not the KirkGPT repository: $RepoRoot"
}

$targets = @(
    "backend\src\llm_followups\rules\engine.py",
    "backend\tests\unit\test_rule_engine.py",
    "backend\tests\integration\test_rule_conversation_api.py"
)

foreach ($relative in $targets) {
    $source = Join-Path $PatchRoot $relative
    $target = Join-Path $RepoRoot $relative

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null

    if (Test-Path $target) {
        Copy-Item $target "$target.finalization.bak" -Force
    }

    Copy-Item $source $target -Force
    Write-Host "wrote $relative"
}

Write-Host ""
Write-Host "Run:"
Write-Host "  cd `"$RepoRoot\backend`""
Write-Host "  pytest tests/unit/test_rule_engine.py -v"
Write-Host "  pytest tests/integration/test_rule_conversation_api.py -v"
Write-Host "  pytest -v"
