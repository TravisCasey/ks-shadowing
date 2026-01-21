"""Custom hatchling build hook for CMake integration."""

import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import (  # ty: ignore[unresolved-import]
    BuildHookInterface,
)


class CMakeBuildHook(BuildHookInterface):
    """Build C++ libraries using CMake and include them in the package."""

    PLUGIN_NAME = "cmake"

    def initialize(self, version: str, build_data: dict) -> None:
        root = Path(self.root)
        build_dir = root / "build"

        subprocess.check_call(["cmake", "-S", ".", "-B", str(build_dir)], cwd=root)
        subprocess.check_call(["cmake", "--build", str(build_dir)], cwd=root)

        # Copy libks2py.so to core/
        ks_so = build_dir / "csrc" / "ksint" / "libks2py.so"
        ks_dest = root / "ks_shadowing" / "core" / "libks2py.so"
        shutil.copy2(ks_so, ks_dest)
        build_data["artifacts"].append("ks_shadowing/core/libks2py.so")

        # Copy libhera2py.so to pha/
        hera_so = build_dir / "csrc" / "hera" / "libhera2py.so"
        hera_dest = root / "ks_shadowing" / "pha" / "libhera2py.so"
        shutil.copy2(hera_so, hera_dest)
        build_data["artifacts"].append("ks_shadowing/pha/libhera2py.so")
