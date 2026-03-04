[CmdletBinding()]
param(
    [string]$Version = "",
    [switch]$SkipPyInstaller
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if (-not $Version) {
    $versionMatch = Select-String -Path "pyproject.toml" -Pattern '^\s*version\s*=\s*"(.*)"\s*$' | Select-Object -First 1
    if (-not $versionMatch) {
        throw "Could not read project version from pyproject.toml."
    }
    $Version = $versionMatch.Matches[0].Groups[1].Value
}

if (-not $SkipPyInstaller) {
    Write-Host "Building PyInstaller onedir bundle..."
    pyinstaller --clean --noconfirm AngioEye.spec
}

$appExe = Join-Path $repoRoot "dist\AngioEye\AngioEye.exe"
if (-not (Test-Path $appExe)) {
    throw "Missing expected executable: $appExe"
}

$isccFromPath = $null
$isccCommand = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if ($isccCommand) {
    $isccFromPath = $isccCommand.Source
}

$isccCandidates = @(@(
    $isccFromPath,
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { $_ -and (Test-Path $_) })

if (-not $isccCandidates) {
    throw "ISCC.exe not found. Install Inno Setup 6 or add ISCC.exe to PATH."
}

$iscc = $isccCandidates[0]
$issPath = Join-Path $repoRoot "installer\AngioEye.iss"

Write-Host "Compiling installer with Inno Setup..."
& $iscc "/DAppVersion=$Version" $issPath
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup compilation failed with exit code $LASTEXITCODE."
}

Write-Host "Installer created in dist\ as AngioEye-setup-$Version.exe"
