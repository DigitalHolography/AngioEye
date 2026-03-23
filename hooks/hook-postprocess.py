from PyInstaller.utils.hooks import collect_submodules

# Ensure dynamically discovered postprocess modules are bundled.
hiddenimports = collect_submodules("postprocess")
