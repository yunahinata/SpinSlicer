# SpinSlicer

SpinSlicer is a local research prototype for tomographic volumetric additive
manufacturing (VAM). It combines a PyQt6 desktop interface, a PyVista/VTK
viewport, a projection generator, a frame player, and an inverse-Radon
reconstruction preview.

The application is intentionally a simulation and projection-preparation tool:
it does not drive a real projector, resin vat, or rotation stage.

## Features

- Load and transform STL models in physical millimetres.
- Generate projection frames with the deterministic internal Radon backend.
- Optionally use the external VAMToolbox CAL optimizer when it is installed.
  The `Auto` mode tries VAMToolbox first and falls back to the internal
  backend, so the default pip environment remains lightweight.
- Create a parametric threaded nut without an STL file. The generator builds a
  watertight ring with a helical triangular internal thread, radial clearance,
  configurable pitch, and configurable sampling density.
- Preview generated frames as a rotating video and export them to MP4.
- Reconstruct an approximate 3D volume with filtered back-projection (FBP).
- Keep every completed run in its own directory with validated metadata and a
  manifest; incomplete runs are not published to the player or simulator.
- English is the default UI language. Russian remains available from the
  language selector and is stored with the application settings.

## Installation

SpinSlicer supports Python 3.11 and 3.12.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

python -m pip install -r requirements.txt -r requirements-dev.txt
```

The package named `pyqtdarktheme` is imported as `qdarktheme`. Video support
uses `opencv-python-headless`.

Run the application with:

```bash
python SpinSlicer.py
```

## Threaded nut / donut printing

Use **Create threaded nut** on the Slicer tab. The dialog exposes:

- outer diameter and nut height;
- maximum threaded-bore diameter;
- thread pitch and radial thread depth;
- radial clearance for a mating part;
- angular and axial sampling density.

The generated mesh is centered and fitted to the current cylindrical vat just
like a loaded STL. It is recorded in `manifest.json` as a parametric
`threaded_nut` model, including all generator parameters. The thread is a
geometrical approximation for volumetric printing experiments; it should be
validated against the intended resin, optical resolution, shrinkage, and
post-cure process before being used as a functional mechanical nut.

For a first print, keep the pitch and thread depth several voxels wide. A
useful rule of thumb is at least 2–3 voxels across the thread depth and a pitch
larger than one voxel. At high resolution, use the VAMToolbox CAL backend to
optimize dose outside the target and improve small positive/negative features.

## Projection backends

The projection engine follows the same conceptual stages as Tomo and
VAMToolbox:

1. voxelize the target geometry;
2. generate or optimize a projection sequence over evenly spaced angles;
3. normalize the dose response and export a frame sequence;
4. reconstruct the expected volume for a visual check.

The internal backend uses a robust mesh-section → rasterization → Radon path
and has no additional installation requirements. The optional VAMToolbox
backend adapts a voxel target to its `TargetGeometry` and parallel-ray
`ProjectionGeometry`, then requests the CAL optimizer. It is not pinned in
`requirements.txt` because the external project uses a separate conda-based
installation and has its own distribution terms. If VAMToolbox is unavailable,
`Auto` reports the fallback and continues with the internal projector.

## Output format

Each completed generation creates a unique run directory:

```text
output_frames/
└── run-20260903T120000Z-a1b2c3d4/
    ├── frame_0000.png
    ├── frame_0001.png
    ├── slice_meta.json
    └── manifest.json
```

`manifest.json` stores the source identity (when the source is an STL), model
type and parameters, transform matrix, projection backend, resin settings,
frame dimensions, and `complete: true`. The frame repository validates image
counts, dimensions, metadata, byte budgets, and completion status before a
player or simulator can consume a run. Legacy flat folders containing
`frame_*.png` remain readable.

## Project layout

| File | Purpose |
| --- | --- |
| `SpinSlicer.py` | Main window, language selector, shared status bar, and log. |
| `slicer_tab.py` | STL/parametric-model workflow and projection generation UI. |
| `nut_dialog.py` | Threaded-nut parameter dialog. |
| `threaded_nut.py` | Watertight helical internal-thread mesh generator. |
| `ui_panels.py` | Process settings and model transform panels. |
| `slicing_engine.py` | Mesh-section rasterization, Radon projection, and export. |
| `vam_backend.py` | Optional VAMToolbox CAL adapter and sinogram layout normalization. |
| `reconstruction.py` | Inverse Radon reconstruction and isosurface preview. |
| `video_tab.py` | Frame playback and MP4 export. |
| `frame_io.py` | Validated frame-set storage, metadata, and manifests. |
| `validation.py` | Mesh/resource preflight and workload budgets. |
| `model_node.py` | Original mesh plus GPU-friendly transform state. |
| `viewport.py` | PyVista/VTK 3D viewport. |
| `workers.py` | Background Qt workers for load, generation, video, and reconstruction. |
| `synthetic_shapes.py` | Canonical numerical-validation phantoms. |
| `tests/` | Geometry, projection, storage, security, and validation tests. |

## Validation and safety

Before generation, SpinSlicer checks mesh finiteness, triangle count, bounds,
vat fit, layer count, estimated memory, frame dimensions, and output budgets.
Long-running work executes in `QThread` workers and supports cancellation.
Output directories are committed only after all PNG files and metadata pass
validation. Local paths are opened through Qt's desktop API rather than a
shell command.

## Development checks

```bash
python -m pytest
python -m ruff check .
python -m mypy
```

GitHub Actions runs the same checks on pushes and pull requests.

## Scientific references

- [Tomo — Print Preparation Software](https://opencal-org.readthedocs.io/en/stable/software/tomo/)
- [VAMToolbox documentation](https://vamtoolbox.readthedocs.io/en/latest/)
- [VAMToolbox source repository](https://github.com/computed-axial-lithography/VAMToolbox)
- [Automatic Exposure Volumetric Additive Manufacturing](https://doi.org/10.1002/admt.202402168)

The DOI paper describes tomographic VAM and automatic exposure control through
real-time scattered-light feedback. SpinSlicer currently implements projection
generation and reconstruction preview only; automatic exposure hardware
feedback is outside the scope of this desktop prototype.
