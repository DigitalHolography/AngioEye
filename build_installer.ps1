<#
.SYNOPSIS
Build the AngioEye Windows installer.

.DESCRIPTION
Builds a PyInstaller one-dir bundle for the GUI application, copies editable
pipeline/postprocess modules next to the bundle, generates an Inno Setup script,
and compiles the final setup executable into dist\installer.

.EXAMPLE
.\build_installer.ps1 -IncludeAllExtras

.EXAMPLE
.\build_installer.ps1 -InnoSetupCompiler "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
#>

[CmdletBinding()]
param(
    [string]$InnoSetupCompiler = "",
    [string]$Python = "",
    [switch]$IncludePipelineExtras,
    [switch]$IncludePostprocessExtras,
    [switch]$IncludeAllExtras,
    [switch]$SkipClean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$PyprojectPath = Join-Path $RepoRoot "pyproject.toml"
$BuildRoot = Join-Path $RepoRoot "build\installer"
$PyInstallerWorkDir = Join-Path $BuildRoot "pyinstaller-work"
$PyInstallerDistDir = Join-Path $BuildRoot "pyinstaller-dist"
$InstallerOutputDir = Join-Path $RepoRoot "dist\installer"
$GeneratedGuiEntryPoint = Join-Path $BuildRoot "angioeye_gui_entry.py"
$GeneratedTclTkRuntimeHook = Join-Path $BuildRoot "angioeye_tcltk_runtime.py"
$GeneratedSpec = Join-Path $BuildRoot "AngioEye.onedir.spec"
$GeneratedInnoScript = Join-Path $BuildRoot "AngioEye.iss"

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Assert-ChildPath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )

    $baseFull = (Get-FullPath $BasePath).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    $targetFull = Get-FullPath $TargetPath
    $prefix = $baseFull + [System.IO.Path]::DirectorySeparatorChar

    if (-not $targetFull.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside the repository: $targetFull"
    }
}

function Read-ProjectMetadata {
    if (-not (Test-Path -LiteralPath $PyprojectPath -PathType Leaf)) {
        throw "pyproject.toml was not found at $PyprojectPath"
    }

    $content = Get-Content -LiteralPath $PyprojectPath -Raw -Encoding UTF8
    $name = "AngioEye"
    $version = $null

    if ($content -match '(?m)^\s*name\s*=\s*"([^"]+)"\s*$') {
        $name = $Matches[1]
    }
    if ($content -match '(?m)^\s*version\s*=\s*"([^"]+)"\s*$') {
        $version = $Matches[1]
    }
    if (-not $version) {
        throw "Could not read [project].version from pyproject.toml"
    }

    return [pscustomobject]@{
        Name = $name
        Version = $version
    }
}

function ConvertTo-InnoQuotedValue {
    param([Parameter(Mandatory = $true)][string]$Value)
    return $Value.Replace('"', '""')
}

function ConvertTo-InnoVersionInfo {
    param([Parameter(Mandatory = $true)][string]$Version)

    if ($Version -notmatch '^\d+(\.\d+){0,3}$') {
        return $null
    }

    $parts = New-Object System.Collections.Generic.List[string]
    foreach ($part in $Version.Split(".")) {
        $parts.Add($part)
    }
    while ($parts.Count -lt 4) {
        $parts.Add("0")
    }
    return ($parts -join ".")
}

function Resolve-InnoSetupCompiler {
    if ($InnoSetupCompiler) {
        if (-not (Test-Path -LiteralPath $InnoSetupCompiler -PathType Leaf)) {
            throw "Inno Setup compiler was not found: $InnoSetupCompiler"
        }
        return (Get-FullPath $InnoSetupCompiler)
    }

    $pathCommand = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($pathCommand) {
        return $pathCommand.Source
    }

    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Get-FullPath $candidate)
        }
    }

    throw "Inno Setup compiler (ISCC.exe) was not found. Install Inno Setup 6 or pass -InnoSetupCompiler."
}

function Resolve-PythonExe {
    if ($Python) {
        if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
            throw "Python executable was not found: $Python"
        }
        return (Get-FullPath $Python)
    }

    $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        return (Get-FullPath $venvPython)
    }

    $pathCommand = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($pathCommand) {
        return $pathCommand.Source
    }

    throw "Python was not found. Install Python, run uv sync, or pass -Python."
}

function Invoke-PyInstaller {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $uv = Get-Command "uv.exe" -ErrorAction SilentlyContinue
    if (-not $uv) {
        $uv = Get-Command "uv" -ErrorAction SilentlyContinue
    }

    if ($uv -and -not $Python) {
        $uvArgs = @("run")
        if ($IncludeAllExtras -or $IncludePipelineExtras) {
            $uvArgs += @("--extra", "pipelines")
        }
        $uvArgs += @("--extra", "postprocess")
        $uvArgs += @("--with", "pyinstaller", "python", "-m", "PyInstaller")
        $uvArgs += $Arguments
        & $uv.Source @uvArgs
        if ($LASTEXITCODE -ne 0) {
            throw "PyInstaller failed with exit code $LASTEXITCODE"
        }
        return
    }

    $pythonExe = Resolve-PythonExe
    & $pythonExe -m PyInstaller @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
}

function Write-GeneratedEntryPoints {
    New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null

    @'
import os
import sys
from pathlib import Path


bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))

for tcl_dir in (
    bundle_root / "_tcl_data",
    bundle_root / "_internal" / "_tcl_data",
    bundle_root / "tcl",
    bundle_root / "tcl8.6",
):
    if (tcl_dir / "init.tcl").is_file():
        os.environ["TCL_LIBRARY"] = str(tcl_dir)
        break

for tk_dir in (
    bundle_root / "_tk_data",
    bundle_root / "_internal" / "_tk_data",
    bundle_root / "tk",
    bundle_root / "tk8.6",
):
    if (tk_dir / "tk.tcl").is_file():
        os.environ["TK_LIBRARY"] = str(tk_dir)
        break

from launcher import main


if __name__ == "__main__":
    main()
'@ | Set-Content -LiteralPath $GeneratedGuiEntryPoint -Encoding UTF8

    @'
import os
import sys
from pathlib import Path


bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))

for tcl_dir in (
    bundle_root / "_tcl_data",
    bundle_root / "_internal" / "_tcl_data",
    bundle_root / "tcl",
    bundle_root / "tcl8.6",
):
    if (tcl_dir / "init.tcl").is_file():
        os.environ["TCL_LIBRARY"] = str(tcl_dir)
        break

for tk_dir in (
    bundle_root / "_tk_data",
    bundle_root / "_internal" / "_tk_data",
    bundle_root / "tk",
    bundle_root / "tk8.6",
):
    if (tk_dir / "tk.tcl").is_file():
        os.environ["TK_LIBRARY"] = str(tk_dir)
        break
'@ | Set-Content -LiteralPath $GeneratedTclTkRuntimeHook -Encoding UTF8
}

function ConvertTo-PythonLiteralPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FullPath $Path).Replace("\", "\\").Replace("'", "\'")
}

function Write-PyInstallerSpec {
    $src = ConvertTo-PythonLiteralPath (Join-Path $RepoRoot "src")
    $guiEntry = ConvertTo-PythonLiteralPath $GeneratedGuiEntryPoint
    $tclTkRuntimeHook = ConvertTo-PythonLiteralPath $GeneratedTclTkRuntimeHook
    $icon = ConvertTo-PythonLiteralPath (Join-Path $RepoRoot "AngioEye.ico")
    $logo = ConvertTo-PythonLiteralPath (Join-Path $RepoRoot "Angioeye_logo.png")
    $settings = ConvertTo-PythonLiteralPath (Join-Path $RepoRoot "default_settings.json")
    $pyproject = ConvertTo-PythonLiteralPath $PyprojectPath

@"
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules


def common_datas():
    datas = []
    datas += collect_data_files("pipelines")
    datas += collect_data_files("postprocess")
    datas += collect_data_files("jinja2")
    datas += collect_data_files("matplotlib")
    datas += collect_data_files("pandas")
    datas += collect_data_files("plotly")
    datas += collect_data_files("scipy")
    datas += collect_data_files("sv_ttk")
    datas += collect_data_files("tkinterdnd2")
    datas += [(r'$logo', ".")]
    datas += [(r'$icon', ".")]
    datas += [(r'$settings', ".")]
    datas += [(r'$pyproject', ".")]
    return datas


def common_hiddenimports():
    hiddenimports = []
    hiddenimports += ["angio_eye", "launcher"]
    hiddenimports += collect_submodules("pipelines")
    hiddenimports += collect_submodules("postprocess")
    hiddenimports += collect_submodules("jinja2")
    hiddenimports += collect_submodules("matplotlib")
    hiddenimports += collect_submodules("pandas")
    hiddenimports += collect_submodules("plotly")
    hiddenimports += collect_submodules("scipy")
    hiddenimports += collect_submodules("tkinterdnd2")
    hiddenimports += ["matplotlib.backends.backend_ps"]
    return hiddenimports


a = Analysis(
    [r'$guiEntry'],
    pathex=[r'$src'],
    binaries=[],
    datas=common_datas(),
    hiddenimports=common_hiddenimports(),
    hookspath=[r'$(ConvertTo-PythonLiteralPath (Join-Path $RepoRoot "hooks"))'],
    hooksconfig={},
    runtime_hooks=[r'$tclTkRuntimeHook'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)


def script_entry(name):
    for item in a.scripts:
        if item[0] == name:
            return [item]
    raise RuntimeError(f"Could not find PyInstaller script entry: {name}")

gui_exe = EXE(
    pyz,
    script_entry("angioeye_gui_entry"),
    [],
    exclude_binaries=True,
    name="AngioEye",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    icon=r'$icon',
)

coll = COLLECT(
    gui_exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AngioEye",
)
"@ | Set-Content -LiteralPath $GeneratedSpec -Encoding UTF8
}

function Copy-EditablePackageModules {
    param(
        [Parameter(Mandatory = $true)][string]$PackageName,
        [Parameter(Mandatory = $true)][string]$BundleDir
    )

    $sourceDir = Join-Path (Join-Path $RepoRoot "src") $PackageName
    $destinationDir = Join-Path $BundleDir $PackageName
    if (-not (Test-Path -LiteralPath $sourceDir -PathType Container)) {
        return
    }

    if (Test-Path -LiteralPath $destinationDir -PathType Container) {
        Remove-Item -LiteralPath $destinationDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null
    Get-ChildItem -LiteralPath $sourceDir -Force |
        Where-Object {
            $_.Name -ne "__pycache__" -and
            $_.Name -ne "__init__.py" -and
            $_.Name -notlike "*.pyc"
        } |
        ForEach-Object {
            $destination = Join-Path $destinationDir $_.Name
            if ($_.PSIsContainer) {
                Copy-Item -LiteralPath $_.FullName -Destination $destination -Recurse -Force
            } else {
                Copy-Item -LiteralPath $_.FullName -Destination $destination -Force
            }
        }
}

function Copy-InstallerExtras {
    param([Parameter(Mandatory = $true)][string]$BundleDir)

    $extraFiles = @(
        "LICENSE",
        "THIRD_PARTY_NOTICES",
        "README.md",
        "AngioEye.ico",
        "default_settings.json",
        "pyproject.toml"
    )

    foreach ($fileName in $extraFiles) {
        $source = Join-Path $RepoRoot $fileName
        if (Test-Path -LiteralPath $source -PathType Leaf) {
            Copy-Item -LiteralPath $source -Destination (Join-Path $BundleDir $fileName) -Force
        }
    }

    Copy-EditablePackageModules -PackageName "pipelines" -BundleDir $BundleDir
    Copy-EditablePackageModules -PackageName "postprocess" -BundleDir $BundleDir
}

function Write-InnoSetupScript {
    param(
        [Parameter(Mandatory = $true)][string]$AppName,
        [Parameter(Mandatory = $true)][string]$AppVersion,
        [Parameter(Mandatory = $true)][string]$BundleDir
    )

    $appNameInno = ConvertTo-InnoQuotedValue $AppName
    $appVersionInno = ConvertTo-InnoQuotedValue $AppVersion
    $bundleDirInno = ConvertTo-InnoQuotedValue (Get-FullPath $BundleDir)
    $outputDirInno = ConvertTo-InnoQuotedValue (Get-FullPath $InstallerOutputDir)
    $licensePathInno = ConvertTo-InnoQuotedValue (Get-FullPath (Join-Path $BundleDir "LICENSE"))
    $iconPathInno = ConvertTo-InnoQuotedValue (Get-FullPath (Join-Path $BundleDir "AngioEye.ico"))
    $setupBaseNameInno = ConvertTo-InnoQuotedValue ("$AppName-setup-$AppVersion")
    $versionInfo = ConvertTo-InnoVersionInfo $AppVersion
    $versionInfoLine = ""
    if ($versionInfo) {
        $versionInfoLine = "VersionInfoVersion=$versionInfo"
    }

    @"
#define MyAppName "$appNameInno"
#define MyAppVersion "$appVersionInno"
#define MyAppPublisher "AngioEye"
#define MyAppExeName "AngioEye.exe"
#define MyBundleDir "$bundleDirInno"
#define MyOutputDir "$outputDirInno"
#define MySetupBaseName "$setupBaseNameInno"
#define MyLicensePath "$licensePathInno"
#define MyIconPath "$iconPathInno"

[Setup]
AppId={{8A8F3E62-62E9-41B5-A3B4-548D203D5A89}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppVerName={#MyAppName} {#MyAppVersion}
DefaultDirName={localappdata}\Programs\{#MyAppName}\{#MyAppVersion}
DefaultGroupName={#MyAppName} {#MyAppVersion}
DisableProgramGroupPage=yes
LicenseFile={#MyLicensePath}
OutputDir={#MyOutputDir}
OutputBaseFilename={#MySetupBaseName}
SetupIconFile={#MyIconPath}
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
$versionInfoLine

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[InstallDelete]
Type: filesandordirs; Name: "{app}\*"

[Files]
Source: "{#MyBundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
"@ | Set-Content -LiteralPath $GeneratedInnoScript -Encoding UTF8
}

$metadata = Read-ProjectMetadata
$appName = $metadata.Name
$appVersion = $metadata.Version
$bundleDir = Join-Path $PyInstallerDistDir "AngioEye"
$guiExe = Join-Path $bundleDir "AngioEye.exe"

Write-Host "Building $appName $appVersion installer..."

if (-not $SkipClean) {
    Assert-ChildPath -BasePath $RepoRoot -TargetPath $BuildRoot
    Assert-ChildPath -BasePath $RepoRoot -TargetPath $InstallerOutputDir
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $InstallerOutputDir -Recurse -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
New-Item -ItemType Directory -Force -Path $InstallerOutputDir | Out-Null
Write-GeneratedEntryPoints
Write-PyInstallerSpec

Invoke-PyInstaller -Arguments @(
    "--noconfirm",
    "--clean",
    "--distpath", $PyInstallerDistDir,
    "--workpath", $PyInstallerWorkDir,
    $GeneratedSpec
)

if (-not (Test-Path -LiteralPath $guiExe -PathType Leaf)) {
    throw "PyInstaller did not produce the expected executable: $guiExe"
}
Copy-InstallerExtras -BundleDir $bundleDir

$iscc = Resolve-InnoSetupCompiler
Write-InnoSetupScript -AppName $appName -AppVersion $appVersion -BundleDir $bundleDir

& $iscc $GeneratedInnoScript
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup failed with exit code $LASTEXITCODE"
}

$installerPath = Join-Path $InstallerOutputDir "$appName-setup-$appVersion.exe"
if (-not (Test-Path -LiteralPath $installerPath -PathType Leaf)) {
    throw "Inno Setup did not produce the expected installer: $installerPath"
}

Write-Host "Installer created: $installerPath"
