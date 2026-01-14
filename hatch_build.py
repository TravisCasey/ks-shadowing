"""Custom hatchling build hook for CMake integration."""

import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import (  # ty: ignore[unresolved-import]
    BuildHookInterface,
)


class CMakeBuildHook(BuildHookInterface):
    """Build the C++ library using CMake and include it in the package."""

    PLUGIN_NAME = "cmake"

    def initialize(self, version: str, build_data: dict) -> None:
        root = Path(self.root)
        build_dir = root / "build"

        subprocess.check_call(["cmake", "-S", ".", "-B", str(build_dir)], cwd=root)
        subprocess.check_call(["cmake", "--build", str(build_dir)], cwd=root)

        # Copy .so to package directory (works for both editable and wheel builds)
        so_path = build_dir / "csrc" / "libks2py.so"
        dest = root / "ks_shadowing" / "libks2py.so"
        shutil.copy2(so_path, dest)

        # Register as build artifact for wheel inclusion
        build_data["artifacts"].append("ks_shadowing/libks2py.so")
