# PowerShell Build Script for ScalarGrad
# Usage: .\build.ps1 [command]
# Commands: clean, build, check, test-upload, upload, install, test, all, release, help

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

function Show-Help {
    Write-Host "ScalarGrad Package Management" -ForegroundColor Cyan
    Write-Host "==============================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host ".\build.ps1 clean        " -NoNewline; Write-Host "- Remove build artifacts" -ForegroundColor Gray
    Write-Host ".\build.ps1 build        " -NoNewline; Write-Host "- Build distribution packages" -ForegroundColor Gray
    Write-Host ".\build.ps1 check        " -NoNewline; Write-Host "- Validate distributions" -ForegroundColor Gray
    Write-Host ".\build.ps1 test-upload  " -NoNewline; Write-Host "- Upload to TestPyPI" -ForegroundColor Gray
    Write-Host ".\build.ps1 upload       " -NoNewline; Write-Host "- Upload to PyPI (production)" -ForegroundColor Gray
    Write-Host ".\build.ps1 install      " -NoNewline; Write-Host "- Install package locally" -ForegroundColor Gray
    Write-Host ".\build.ps1 test         " -NoNewline; Write-Host "- Run test suite" -ForegroundColor Gray
    Write-Host ".\build.ps1 all          " -NoNewline; Write-Host "- Clean, build, check, upload to TestPyPI" -ForegroundColor Gray
    Write-Host ".\build.ps1 release      " -NoNewline; Write-Host "- Clean, build, check, upload to PyPI" -ForegroundColor Gray
    Write-Host ""
}

function Invoke-Clean {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build, dist, *.egg-info
    Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
    Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
    Write-Host "[OK] Clean complete" -ForegroundColor Green
}

function Invoke-Build {
    Write-Host "Building distribution packages..." -ForegroundColor Yellow
    Invoke-Clean
    python -m build
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Build complete" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Build failed" -ForegroundColor Red
        exit 1
    }
}

function Invoke-Check {
    Write-Host "Validating distributions..." -ForegroundColor Yellow
    if (-not (Test-Path "dist")) {
        Invoke-Build
    }
    twine check dist/*
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Check complete" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Check failed" -ForegroundColor Red
        exit 1
    }
}

function Invoke-TestUpload {
    Write-Host "Uploading to TestPyPI..." -ForegroundColor Yellow
    Invoke-Check
    twine upload --repository testpypi dist/* --verbose
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Uploaded to TestPyPI" -ForegroundColor Green
        Write-Host "Install with: pip install --index-url https://test.pypi.org/simple/ scalargrad" -ForegroundColor Cyan
    } else {
        Write-Host "[ERROR] Upload failed" -ForegroundColor Red
        exit 1
    }
}

function Invoke-Upload {
    Write-Host "WARNING: This will upload to production PyPI!" -ForegroundColor Red
    $confirm = Read-Host "Are you sure? (yes/no)"
    if ($confirm -ne "yes") {
        Write-Host "Upload cancelled" -ForegroundColor Yellow
        exit 0
    }
    
    Write-Host "Uploading to PyPI..." -ForegroundColor Yellow
    Invoke-Check
    twine upload dist/* --verbose
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Uploaded to PyPI" -ForegroundColor Green
        Write-Host "Install with: pip install scalargrad" -ForegroundColor Cyan
    } else {
        Write-Host "[ERROR] Upload failed" -ForegroundColor Red
        exit 1
    }
}

function Invoke-Install {
    Write-Host "Installing package in editable mode..." -ForegroundColor Yellow
    pip install -e .
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Package installed" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Installation failed" -ForegroundColor Red
        exit 1
    }
}

function Invoke-Test {
    Write-Host "Running test suite..." -ForegroundColor Yellow
    pytest -v
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Tests complete" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Tests failed" -ForegroundColor Red
        exit 1
    }
}

function Invoke-All {
    Write-Host "Running all steps (TestPyPI)..." -ForegroundColor Cyan
    Invoke-Clean
    Invoke-Build
    Invoke-Check
    Invoke-TestUpload
    Write-Host "[OK] All steps complete" -ForegroundColor Green
}

function Invoke-Release {
    Write-Host "Running release steps (PyPI)..." -ForegroundColor Cyan
    Invoke-Clean
    Invoke-Build
    Invoke-Check
    Invoke-Upload
    Write-Host "[OK] Release complete" -ForegroundColor Green
}

# Main command dispatcher
switch ($Command.ToLower()) {
    "clean"       { Invoke-Clean }
    "build"       { Invoke-Build }
    "check"       { Invoke-Check }
    "test-upload" { Invoke-TestUpload }
    "upload"      { Invoke-Upload }
    "install"     { Invoke-Install }
    "test"        { Invoke-Test }
    "all"         { Invoke-All }
    "release"     { Invoke-Release }
    "help"        { Show-Help }
    default       { 
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host ""
        Show-Help
        exit 1
    }
}
