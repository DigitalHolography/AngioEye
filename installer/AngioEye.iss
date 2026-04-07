#define MyAppName "AngioEye"

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

#ifndef PayloadDir
  #error PayloadDir must be provided on the ISCC command line.
#endif

#ifndef OutputDir
  #define OutputDir "dist"
#endif

[Setup]
AppId={{08E860C9-5027-4E3E-99C5-E7D9F4A58216}
AppName={#MyAppName}
AppVersion={#AppVersion}
AppVerName={#MyAppName} {#AppVersion}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile={#PayloadDir}\LICENSE
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=AngioEye-setup-{#AppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
SetupIconFile={#PayloadDir}\AngioEye.ico
UninstallDisplayIcon={app}\AngioEye.exe

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\AngioEye.exe"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\AngioEye.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\AngioEye.exe"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
