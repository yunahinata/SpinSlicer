from __future__ import annotations

from PIL import Image

from frame_io import GenerationManifest, SliceMeta, save_manifest, save_meta
from virtual_device import PlaybackCancelled, VirtualPrinter


def _write_run(tmp_path):
    run_dir = tmp_path / "run-virtual"
    run_dir.mkdir()
    for index in range(3):
        Image.new("L", (8, 6), color=index).save(run_dir / f"frame_{index:04d}.png")
    save_meta(
        str(run_dir),
        SliceMeta(
            diameter_mm=60.0,
            grid_res=16,
            output_res=32,
            num_frames=3,
            fill_holes=True,
            resin_base_exposure=1.0,
            resin_intensity=100.0,
            resin_threshold=0.0,
        ),
    )
    save_manifest(
        str(run_dir),
        GenerationManifest(
            frame_count=3,
            frame_size=(8, 6),
            machine_profile={"name": "test-machine"},
            resin_profile={"name": "test-resin"},
            frame_schedule={
                "angles_deg": [10.0, 130.0, 250.0],
                "frame_duration_s": 0.25,
            },
            complete=True,
        ),
    )
    return run_dir


def test_virtual_printer_replays_manifest_schedule(tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    seen = []

    report = VirtualPrinter().play(str(run_dir), on_event=seen.append)

    assert report.frame_count == 3
    assert report.total_duration_s == 0.75
    assert [event.index for event in seen] == [0, 1, 2]
    assert [event.angle_deg for event in seen] == [10.0, 130.0, 250.0]
    assert all(len(event.sha256) == 64 for event in seen)


def test_virtual_printer_can_cancel(tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    calls = 0

    def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    try:
        VirtualPrinter().play(str(run_dir), is_cancelled=cancelled)
    except PlaybackCancelled:
        pass
    else:
        raise AssertionError("virtual playback did not cancel")
