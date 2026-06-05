param(
  [string]$HostIp = "192.168.1.5",
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($conn in $listener) {
  Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
}

$env:EXPO_PUBLIC_API_BASE_URL = "http://${HostIp}:${Port}"
$python = Resolve-Path ".\.venv\Scripts\python.exe"

Start-Process `
  -FilePath $python `
  -ArgumentList @("-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "$Port") `
  -WorkingDirectory $RepoRoot `
  -RedirectStandardOutput ".\server\backend_stdout.log" `
  -RedirectStandardError ".\server\backend_stderr.log" `
  -WindowStyle Hidden

Write-Host "Waiting for backend health on http://127.0.0.1:${Port}/health ..."
for ($attempt = 1; $attempt -le 60; $attempt++) {
  try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:${Port}/health" -TimeoutSec 5
    $health | ConvertTo-Json -Depth 6
    Write-Host "Backend health OK after ${attempt}s."
    exit 0
  } catch {
    Start-Sleep -Seconds 1
  }
}

Write-Error "Backend did not become healthy within 60 seconds."
Write-Host "`n--- server/backend_stdout.log ---"
Get-Content ".\server\backend_stdout.log" -Tail 120 -ErrorAction SilentlyContinue
Write-Host "`n--- server/backend_stderr.log ---"
Get-Content ".\server\backend_stderr.log" -Tail 120 -ErrorAction SilentlyContinue
exit 1
