$ErrorActionPreference = "Stop"

$listeners = Get-NetTCPConnection `
    -LocalAddress 127.0.0.1 `
    -LocalPort 8000 `
    -State Listen `
    -ErrorAction SilentlyContinue

foreach ($listener in $listeners) {
    $process = Get-CimInstance Win32_Process `
        -Filter "ProcessId = $($listener.OwningProcess)"
    if ($process.CommandLine -like "*medical_agent.web*") {
        Stop-Process -Id $listener.OwningProcess
    } else {
        throw "Port 8000 is in use by another application."
    }
}

& (Join-Path $PSScriptRoot "start-web.ps1")
