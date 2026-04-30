# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

def _common_datas():
    datas = []
    datas += collect_data_files("pipelines")
    datas += collect_data_files("postprocess")
    datas += collect_data_files("sv_ttk")
    datas += collect_data_files("tkinterdnd2")
    datas += [("Angioeye_logo.png", ".")]
    datas += [("AngioEye.ico", ".")]
    datas += [("default_settings.json", ".")]
    datas += [("pyproject.toml", ".")]
    return datas


def _common_hiddenimports():
    hiddenimports = []
    hiddenimports += collect_submodules("pipelines")
    hiddenimports += collect_submodules("postprocess")
    hiddenimports += collect_submodules("tkinterdnd2")
    hiddenimports += ["matplotlib.backends.backend_ps"]
    return hiddenimports


def _build_exe(script_path, name, console):
    analysis = Analysis(
        [script_path],
        pathex=["src"],
        binaries=[],
        datas=_common_datas(),
        hiddenimports=_common_hiddenimports(),
        hookspath=["hooks"],
        hooksconfig={},
        runtime_hooks=[],
        excludes=[],
        noarchive=False,
        optimize=0,
    )
    pyz = PYZ(analysis.pure)
    return EXE(
        pyz,
        analysis.scripts,
        analysis.binaries,
        analysis.datas,
        [],
        name=name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=console,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon="AngioEye.ico",
    )


gui_exe = _build_exe("src\\angio_eye.py", "AngioEye", False)
cli_exe = _build_exe("src\\cli.py", "AngioEyeCLI", True)
