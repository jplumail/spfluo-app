from __future__ import annotations

import os
import subprocess
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name == "wheel":
            # Build JAR and adds it in wheel
            here = os.path.abspath(os.path.dirname(__file__))
            tipi_path = os.path.join(here, "singleparticle", "_vendored", "TiPi")
            subprocess.check_call(
                ["mvn", "package", "--file", "pom.xml", "--settings", "settings.xml"],
                cwd=tipi_path,
            )
            jar_file = os.path.join(tipi_path, "target", "TiPi-for-spfluo-1.0.jar")
            assert os.path.exists(jar_file), f"JAR file {jar_file} not found"

            # Build parcel
            web_path = os.path.join(here, "singleparticle", "web", "client")
            subprocess.check_call(["npm", "ci", "--include", "dev"], cwd=web_path)
            subprocess.check_call(["npx", "parcel", "build", "src/*"], cwd=web_path)
