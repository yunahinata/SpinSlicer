"""Compile the portable native SIRT kernel for the current build platform."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "native" / "sirt_engine.cpp"
OUTPUT_DIR = PROJECT_ROOT / "build" / "native"


def _compiler_command() -> list[str]:
    configured = os.environ.get("CXX", "").strip()
    if configured:
        command = shlex.split(configured)
        if command:
            return command

    compiler_names = ("cl.exe", "cl") if sys.platform == "win32" else ("g++", "c++")
    for name in compiler_names:
        compiler = shutil.which(name)
        if compiler:
            return [compiler]
    raise RuntimeError(
        "A C++ compiler was not found. On Windows, run this from a Visual Studio "
        "Developer PowerShell; on Linux, install g++ or set CXX."
    )


def build() -> Path:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Native SIRT source is missing: {SOURCE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    compiler = _compiler_command()

    compiler_name = Path(compiler[0]).name.casefold()
    is_msvc = compiler_name in {"cl", "cl.exe"}

    if sys.platform == "win32" and is_msvc:
        output_path = OUTPUT_DIR / "libsirt.dll"
        object_path = OUTPUT_DIR / "sirt_engine.obj"
        import_library = OUTPUT_DIR / "libsirt.lib"
        command = compiler + [
            "/nologo",
            "/LD",
            "/O2",
            "/EHsc",
            "/std:c++17",
            "/openmp:experimental",
            str(SOURCE),
            f"/Fo{object_path}",
            "/link",
            f"/OUT:{output_path}",
            f"/IMPLIB:{import_library}",
        ]
    elif sys.platform.startswith("linux"):
        output_path = OUTPUT_DIR / "libsirt.so"
        command = compiler + [
            "-shared",
            "-fPIC",
            "-O3",
            "-std=c++17",
            "-fopenmp",
            str(SOURCE),
            "-o",
            str(output_path),
        ]
    elif sys.platform == "win32":
        output_path = OUTPUT_DIR / "libsirt.dll"
        command = compiler + [
            "-shared",
            "-O3",
            "-std=c++17",
            "-fopenmp",
            str(SOURCE),
            "-o",
            str(output_path),
        ]
    else:
        raise RuntimeError(f"Native SIRT build is not configured for {sys.platform}.")

    subprocess.run(command, cwd=OUTPUT_DIR, check=True)
    if not output_path.is_file():
        raise RuntimeError(f"The compiler did not produce {output_path}.")
    print(f"Built native SIRT library: {output_path}")
    return output_path


if __name__ == "__main__":
    build()
