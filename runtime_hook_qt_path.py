"""Keep PyQt6 bound to the Qt DLLs bundled inside the one-file executable."""

import os
import sys

_dll_directory_handles: list[object] = []

if sys.platform.startswith("win") and hasattr(sys, "_MEIPASS"):
    _bundle_root = os.path.abspath(getattr(sys, "_MEIPASS"))
    _qt_bin = os.path.join(_bundle_root, "PyQt6", "Qt6", "bin")
    for _directory in (_qt_bin, _bundle_root):
        if not os.path.isdir(_directory):
            continue
        _dll_directory = getattr(os, "add_dll_directory", None)
        if callable(_dll_directory):
            _dll_directory_handles.append(_dll_directory(_directory))
        os.environ["PATH"] = _directory + os.pathsep + os.environ.get("PATH", "")
