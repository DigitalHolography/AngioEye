from PyInstaller.utils.hooks import collect_submodules

# Ensure dynamically discovered pipeline modules are bundled.
hiddenimports = collect_submodules("pipelines")
