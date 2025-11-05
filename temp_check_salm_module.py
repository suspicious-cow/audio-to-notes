"""Quick helper script to inspect installed NeMo speechlm2 modules.
Delete once finished debugging.
"""

import pkgutil
import sys
from pathlib import Path

site_packages = Path(sys.prefix) / "Lib" / "site-packages"
base_dir = site_packages / "nemo" / "collections" / "speechlm2"
models_dir = base_dir / "models"

print(f"Looking in: {models_dir}")

if not models_dir.exists():
    print("speechlm2 models directory not found.")
    sys.exit(1)

available = sorted(m.name for m in pkgutil.iter_modules([models_dir.as_posix()]))
print("Available model modules:")
for name in available:
    print(f"- {name}")

loader = pkgutil.get_loader("nemo.collections.speechlm2")
print(f"Loader for nemo.collections.speechlm2: {loader}")
