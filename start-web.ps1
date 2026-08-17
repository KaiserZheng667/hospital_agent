$ErrorActionPreference = "Stop"

$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$bundledPython = "C:\Users\Mordekaiser\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$legacyPackages = Join-Path $projectDirectory ".venv\Lib\site-packages"
$projectPython = Join-Path $projectDirectory ".venv\Scripts\python.exe"

Set-Location $projectDirectory

if (Test-Path $projectPython) {
    & $projectPython -c "import fastapi, langgraph, uvicorn" 2>$null
    if ($LASTEXITCODE -eq 0) {
        & $projectPython -m medical_agent.web
        exit $LASTEXITCODE
    }
}

if (-not (Test-Path $bundledPython)) {
    throw "No usable Python runtime was found."
}

$env:PYTHONPATH = "$projectDirectory\src;$legacyPackages"
& $bundledPython -m medical_agent.web
