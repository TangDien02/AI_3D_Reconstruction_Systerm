param(
    [string]$RepoDir = "external\TripoSR"
)

$ErrorActionPreference = "Stop"

$PyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([version]$PyVersion -ge [version]"3.13") {
    throw "TripoSR dependencies are pinned for Python 3.8-3.12. Create a Python 3.10/3.11 venv first, for example: py -3.10 -m venv .venv-triposr"
}

python -m pip install --upgrade pip
python -m pip install "setuptools<82,>=70.2.0"
python -m pip install -r requirements.txt
python -m pip install -r requirements-triposr.txt

if (Test-Path $RepoDir) {
    git -C $RepoDir pull --ff-only
} else {
    New-Item -ItemType Directory -Force -Path (Split-Path $RepoDir) | Out-Null
    git clone https://github.com/VAST-AI-Research/TripoSR.git $RepoDir
}

Write-Host "TripoSR core runtime is ready."
Write-Host "Repo dir: $RepoDir"
