from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name == "wheel":
            # Build JAR and adds it in wheel
            here = os.path.abspath(os.path.dirname(__file__))
            tipi_path = os.path.join(here, "singleparticle", "_vendored", "TiPi")
            mvn_executable = "mvn.cmd" if sys.platform == "win32" else "mvn"
            try:
                subprocess.check_call(
                    [mvn_executable, "package", "--file", "pom.xml", "--settings", "settings.xml"],
                    cwd=tipi_path,
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find Maven executable '{mvn_executable}'. "
                    "Please make sure Maven is installed and available in the PATH."
                )
            jar_file = os.path.join(tipi_path, "target", "TiPi-for-spfluo-1.0.jar")
            assert os.path.exists(jar_file), f"JAR file {jar_file} not found"

            # Build parcel
            web_path = os.path.join(here, "singleparticle", "web", "client")
            npm_executable = "npm.cmd" if sys.platform == "win32" else "npm"
            npx_executable = "npx.cmd" if sys.platform == "win32" else "npx"
            try:
                subprocess.check_call([npm_executable, "ci", "--include", "dev"], cwd=web_path)
                subprocess.check_call([npx_executable, "parcel", "build", "src/*"], cwd=web_path)
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Could not find npm or npx executables. "
                    "Please make sure Node.js is installed and available in the PATH."
                )
