; Inno Setup script for packaging the PyInstaller onedir build.
; Compile with:
;   ISCC.exe /DAppVersion=0.1.0 installer\AngioEye.iss

#define MyAppName "AngioEye"
#define MyAppExeName "AngioEye.exe"
#define MyAppPublisher "AngioEye"

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{2A83EEA6-B9B4-4B6F-87C8-22E504B8B109}
AppName={#MyAppName}
AppVersion={#AppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
OutputDir=..\dist
OutputBaseFilename={#MyAppName}-setup-{#AppVersion}
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\AngioEye\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
