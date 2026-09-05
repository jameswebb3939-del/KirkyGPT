param(
    [string]$RepoRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"

$PatchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path $RepoRoot).Path

if (-not (Test-Path (Join-Path $RepoRoot ".git"))) {
    throw "RepoRoot must point to the KirkGPT Git repository root."
}

Write-Host "Applying KirkGPT normalization patch to $RepoRoot"

# Remove/rename old native include directory so Linux builds do not retain
# a case-sensitive duplicate.
$oldNativeDirs = @(
    "backend/native/include/ec_pro_native",
    "backend/native/include/KirkGPT_native"
)

$newNativeDir = Join-Path $RepoRoot "backend/native/include/kirk_gpt_native"

foreach ($relative in $oldNativeDirs) {
    $old = Join-Path $RepoRoot $relative
    if (Test-Path $old) {
        if (Test-Path $newNativeDir) {
            Remove-Item -Recurse -Force $old
        } else {
            New-Item -ItemType Directory -Force (Split-Path -Parent $newNativeDir) | Out-Null
            Move-Item $old $newNativeDir
        }
    }
}

# Copy every normalized patch file over its repository counterpart.
Get-ChildItem -Path $PatchRoot -Recurse -File |
    Where-Object {
        $_.Name -ne "apply_kirkgpt_normalization.ps1" -and
        $_.Name -ne "README_PATCH.txt"
    } |
    ForEach-Object {
        $relative = $_.FullName.Substring($PatchRoot.Length).TrimStart("\","/")
        $target = Join-Path $RepoRoot $relative
        New-Item -ItemType Directory -Force (Split-Path -Parent $target) | Out-Null
        Copy-Item $_.FullName $target -Force
    }

# Preserve an existing SQLite database while normalizing its filename.
$dbCandidates = @(
    "backend/data/ec_pro.db",
    "backend/data/KirkGPT.db"
)
$newDb = Join-Path $RepoRoot "backend/data/kirk_gpt.db"

foreach ($relative in $dbCandidates) {
    $oldDb = Join-Path $RepoRoot $relative
    if ((Test-Path $oldDb) -and -not (Test-Path $newDb)) {
        Move-Item $oldDb $newDb
        break
    }
}

Write-Host ""
Write-Host "Legacy EC Pro search:"
Push-Location $RepoRoot
try {
    git grep -n -I -E "EC Pro|ECPro|EC_PRO|ec_pro|ec-pro"
    if ($LASTEXITCODE -eq 1) {
        Write-Host "  No tracked legacy EC Pro references found."
    }

    Write-Host ""
    Write-Host "Git status:"
    git status --short
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Patch applied. Rebuild the native extension from a clean build directory before testing."
