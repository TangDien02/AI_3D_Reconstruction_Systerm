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

Start-Sleep -Seconds 6
Invoke-RestMethod -Uri "http://127.0.0.1:${Port}/health" | ConvertTo-Json -Depth 6
