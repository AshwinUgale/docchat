# Full check pipeline for DocChat.
#
# Runs both halves of the codebase in one invocation:
#   - extension/  (TypeScript: tsc --noEmit + ESLint)
#   - sidecar/    (Python:    ruff + mypy strict + pytest)
#
# Fail-fast: any step that fails aborts the pipeline with a non-zero exit.
# Use this before every commit.

$ErrorActionPreference = "Stop"

function Step($Name, $Cmd) {
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAIL: $Name (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

$start = Get-Date

# ---------------------------------------------------------------------------
# Python sidecar
# ---------------------------------------------------------------------------
Push-Location sidecar
try {
    Step "ruff check --fix" { uv run ruff check . --fix }
    Step "ruff format"      { uv run ruff format . }
    Step "mypy strict"      { uv run mypy src }
    Step "pytest"           { uv run pytest }
}
finally {
    Pop-Location
}

# ---------------------------------------------------------------------------
# TypeScript extension
# ---------------------------------------------------------------------------
Push-Location extension
try {
    Step "tsc --noEmit"     { npx --no-install tsc --noEmit }
    # ESLint added when src grows; placeholder for now
    if (Test-Path .eslintrc.cjs) {
        Step "eslint"       { npx --no-install eslint src --max-warnings=0 }
    }
}
finally {
    Pop-Location
}

$elapsed = [int]((Get-Date) - $start).TotalSeconds
Write-Host "==> All checks passed in ${elapsed}s." -ForegroundColor Green
