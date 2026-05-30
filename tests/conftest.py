import sys

from pathlib import Path

# This dynamically finds the scripts folder
scripts_dir = str(Path(__file__).parent.parent / "scripts")

# Injects the scripts folder into the Python path so 'from <sut> import...' works
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
