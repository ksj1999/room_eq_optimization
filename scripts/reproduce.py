"""Verify the local MeshRIR archive and regenerate all experiment results."""

import subprocess
import sys
from pathlib import Path

from room_eq.data import ARCHIVE, verify_archive


def main():
    project = Path(__file__).resolve().parents[1]
    verify_archive(project / ARCHIVE)
    for module in ("smoke", "experiments", "ablation", "peak_ablation"):
        subprocess.run([sys.executable, "-m", f"room_eq.{module}"], cwd=project, check=True)


if __name__ == "__main__":
    main()
