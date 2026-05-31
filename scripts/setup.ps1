param(
    [string]$VenvPath = ".venv",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Find-Python311 {
    $candidates = @(
        @{ Label = "py -3.11"; Command = { py -3.11 -c "import sys; print(sys.executable)" } },
        @{ Label = "python3.11"; Command = { python3.11 -c "import sys; print(sys.executable)" } },
        @{ Label = "python"; Command = { python -c "import sys; print(sys.executable)" } }
    )

    foreach ($candidate in $candidates) {
        try {
            $resolved = (& $candidate.Command).Trim()
            if (-not $resolved) {
                continue
            }

            $version = (& $resolved -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
            if ($version -eq "3.11") {
                return $resolved
            }
        }
        catch {
            continue
        }
    }

    throw "Python 3.11 was not found. Install Python 3.11.x, then run scripts\setup.ps1 again."
}

$pythonExe = Find-Python311
Write-Host "Using Python:" $pythonExe

if (Test-Path -LiteralPath $VenvPath) {
    Remove-Item -LiteralPath $VenvPath -Recurse -Force
}

& $pythonExe -m venv $VenvPath
$venvPython = Join-Path $VenvPath "Scripts\python.exe"

& $venvPython -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"
& $venvPython -m pip install --upgrade pip

if (-not $SkipInstall) {
    & $venvPython -m pip install -r requirements.txt
}

& $venvPython scripts\verify_runtime.py

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate with: $VenvPath\Scripts\Activate.ps1"
Write-Host "Run backend with: $venvPython -m uvicorn server.main:app --host 0.0.0.0 --port 8000"
