# Hardware Delta-Z Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested Windows secondary-monitor SLM and Allied Vision camera loop that measures experimental image quality for every independently optimized `delta_z`.

**Architecture:** Keep `delta_z_scan.py` as the offline WGS generator and add `experimental_delta_z_scan.py` as a dependency-injected acquisition and analysis module. Real `wxPython` and `vmbpy` imports remain lazy; unit and integration tests use in-memory hardware doubles.

**Tech Stack:** Python 3, NumPy, SciPy, Pillow, Matplotlib, wxPython/slmpy, Allied Vision Vimba X/vmbpy, pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-hardware-delta-z-loop-design.md`

## Global Constraints

- Support Windows secondary-monitor SLM control through the existing `slmpy` path.
- Support Allied Vision acquisition through the existing `avt.VimbaCamera` wrapper.
- Never interpolate the 1272 x 1024 physical phase image.
- Keep SLM, objective relay, imaging objective, and CCD mechanically fixed during the scan.
- Preserve the independent-WGS-per-`delta_z` behavior in `delta_z_scan.py`.
- Do not import `wxPython` or `vmbpy` during offline analysis or unit tests.
- Do not silently overwrite a prior experimental acquisition.

---

### Task 1: Scan Manifest, Calibration, and Exact SLM Transport

**Files:**
- Create: `experimental_delta_z_scan.py`
- Create: `tests/test_experimental_delta_z_scan.py`

**Interfaces:**
- Produces: `ScanPoint(delta_z_mm: float, bmp_path: Path, scan_label: str)`.
- Produces: `load_scan_points(scan_dir: Path) -> list[ScanPoint]`.
- Produces: `apply_slm_calibration(phase: np.ndarray, correction: np.ndarray | None, lut: int) -> np.ndarray`.
- Produces: `center_phase_on_display(phase: np.ndarray, display_size_xy: tuple[int, int]) -> np.ndarray`.

- [ ] **Step 1: Write failing manifest and pixel-mapping tests**

```python
def test_center_phase_on_1280_transport_without_interpolation():
    phase = np.arange(1024 * 1272, dtype=np.uint8).reshape(1024, 1272)
    frame = scan.center_phase_on_display(phase, (1280, 1024))
    assert frame.shape == (1024, 1280)
    np.testing.assert_array_equal(frame[:, 4:1276], phase)
    assert not frame[:, :4].any()
    assert not frame[:, 1276:].any()

def test_load_scan_points_uses_csv_order_and_requires_every_bmp(tmp_path):
    names = [
        "slm_phase_scan_000_delta_z_m005.000mm.bmp",
        "slm_phase_scan_001_delta_z_p000.000mm.bmp",
    ]
    for name in names:
        Image.fromarray(np.zeros((4, 6), np.uint8)).save(tmp_path / name)
    with (tmp_path / "delta_z_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["delta_z_mm", "bmp_file"])
        writer.writeheader()
        writer.writerow({"delta_z_mm": -5, "bmp_file": names[0]})
        writer.writerow({"delta_z_mm": 0, "bmp_file": names[1]})
    points = scan.load_scan_points(tmp_path)
    assert [point.bmp_path.name for point in points] == names
    (tmp_path / names[1]).unlink()
    with pytest.raises(FileNotFoundError, match=names[1]):
        scan.load_scan_points(tmp_path)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `/home/public/zcq/ccbc/slm_code/.venv/bin/python -m pytest tests/test_experimental_delta_z_scan.py -v`

Expected: collection fails because `experimental_delta_z_scan` does not exist.

- [ ] **Step 3: Implement the minimal pure functions**

```python
@dataclass(frozen=True)
class ScanPoint:
    delta_z_mm: float
    bmp_path: Path
    scan_label: str

def center_phase_on_display(phase, display_size_xy):
    display_width, display_height = display_size_xy
    height, width = phase.shape
    if width > display_width or height > display_height:
        raise ValueError("SLM phase does not fit the display transport")
    frame = np.zeros((display_height, display_width), dtype=np.uint8)
    x0 = (display_width - width) // 2
    y0 = (display_height - height) // 2
    frame[y0:y0 + height, x0:x0 + width] = phase
    return frame
```

- [ ] **Step 4: Run focused and complete tests**

Run: `/home/public/zcq/ccbc/slm_code/.venv/bin/python -m pytest tests/test_experimental_delta_z_scan.py -v`

Run: `/home/public/zcq/ccbc/slm_code/.venv/bin/python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the pure SLM preparation layer**

```bash
git add experimental_delta_z_scan.py tests/test_experimental_delta_z_scan.py
git commit -m "feat: prepare ordered delta-z frames for SLM display"
```

### Task 2: Dependency-Injected Acquisition Loop

**Files:**
- Modify: `experimental_delta_z_scan.py`
- Modify: `tests/test_experimental_delta_z_scan.py`

**Interfaces:**
- Consumes: `ScanPoint`, `apply_slm_calibration`, and `center_phase_on_display` from Task 1.
- Produces: `AcquiredPoint(point: ScanPoint, average: np.ndarray, saturation_fraction: float, peak_raw: float)`.
- Produces: `acquire_scan_points(points, display, camera, output_dir, *, exposure_us, frames_per_point, settle_seconds, correction, lut, save_raw_frames, sleep_fn=time.sleep) -> list[AcquiredPoint]`.

- [ ] **Step 1: Write a failing end-to-end acquisition test with in-memory hardware**

```python
def test_acquisition_displays_in_order_averages_frames_and_saves_outputs(tmp_path):
    display = FakeDisplay(size=(8, 4))
    camera = FakeCamera([np.ones((6, 6)), np.full((6, 6), 3)] * 2)
    acquired = scan.acquire_scan_points(
        points, display, camera, tmp_path / "experiment",
        exposure_us=50, frames_per_point=2, settle_seconds=0,
        correction=None, lut=256, save_raw_frames=True, sleep_fn=lambda _: None,
    )
    assert [frame[0, 1] for frame in display.frames] == [first_value, second_value]
    assert np.all(acquired[0].average == 2)
    assert camera.exposures == [50, 50, 50, 50]
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `/home/public/zcq/ccbc/slm_code/.venv/bin/python -m pytest tests/test_experimental_delta_z_scan.py::test_acquisition_displays_in_order_averages_frames_and_saves_outputs -v`

Expected: FAIL because `acquire_scan_points` is missing.

- [ ] **Step 3: Implement acquisition, validation, output protection, and file saving**

```python
for point in points:
    display.updateArray(center_phase_on_display(prepared, display.getSize()))
    sleep_fn(settle_seconds)
    stack = np.stack([camera.capture(exposure_us) for _ in range(frames_per_point)])
    average = stack.astype(np.float64).mean(axis=0)
    np.save(output_dir / f"camera_average_{point.scan_label}.npy", average)
```

- [ ] **Step 4: Add and pass failure-path tests**

```python
def test_acquisition_rejects_mismatched_camera_shapes(tmp_path):
    camera = FakeCamera([np.zeros((4, 4)), np.zeros((5, 4))])
    with pytest.raises(ValueError, match="camera frame shape changed"):
        scan.acquire_scan_points(
            points[:1], FakeDisplay((8, 4)), camera, tmp_path / "experiment",
            exposure_us=50, frames_per_point=2, settle_seconds=0,
            correction=None, lut=256, save_raw_frames=False,
            sleep_fn=lambda _: None,
        )

@pytest.mark.parametrize("frames", [0, -1])
def test_acquisition_rejects_nonpositive_frame_count(tmp_path, frames):
    with pytest.raises(ValueError, match="frames_per_point must be positive"):
        scan.acquire_scan_points(
            points, FakeDisplay((8, 4)), FakeCamera([]), tmp_path / "experiment",
            exposure_us=50, frames_per_point=frames, settle_seconds=0,
            correction=None, lut=256, save_raw_frames=False,
        )

def test_acquisition_refuses_nonempty_output_directory(tmp_path):
    output = tmp_path / "experiment"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError, match="fresh output directory"):
        scan.acquire_scan_points(
            points, FakeDisplay((8, 4)), FakeCamera([]), output,
            exposure_us=50, frames_per_point=1, settle_seconds=0,
            correction=None, lut=256, save_raw_frames=False,
        )
    assert marker.read_text() == "keep"
```

Verify each exception is raised before unrelated files are overwritten.

Run: `/home/public/zcq/ccbc/slm_code/.venv/bin/python -m pytest tests/test_experimental_delta_z_scan.py -v`

Expected: all acquisition tests pass.

- [ ] **Step 5: Commit the acquisition loop**

```bash
git add experimental_delta_z_scan.py tests/test_experimental_delta_z_scan.py
git commit -m "feat: acquire AVT frames for each delta-z phase"
```

### Task 3: Stable Spot Detection and Experimental Quality Metrics

**Files:**
- Modify: `experimental_delta_z_scan.py`
- Modify: `tests/test_experimental_delta_z_scan.py`

**Interfaces:**
- Consumes: `AcquiredPoint` from Task 2.
- Produces: `detect_common_spots(images, expected_count, *, roi_xywh, min_peak_distance_px, background_percentile) -> np.ndarray` with `(y, x)` centers.
- Produces: `analyze_acquired_points(acquired, centers, *, spot_radius_px, background_percentile) -> list[dict[str, float | str]]`.
- Produces: `rank_quality(rows) -> tuple[list[dict[str, float | str]], dict[str, float | str]]`.

- [ ] **Step 1: Write failing synthetic-image tests**

```python
def test_common_spots_and_metrics_prefer_sharp_uniform_low_halo_image():
    sharp = synthetic_grid(sigmas=[1, 1, 1, 1], amplitudes=[100] * 4)
    blurred = synthetic_grid(sigmas=[3, 3, 3, 3], amplitudes=[100, 80, 60, 40]) + 5
    centers = scan.detect_common_spots([sharp, blurred], 4, min_peak_distance_px=8)
    rows, best = scan.analyze_and_rank(acquired_points, centers, spot_radius_px=6)
    assert rows[0]["mean_fwhm_px"] < rows[1]["mean_fwhm_px"]
    assert rows[0]["target_plane_uniformity"] > rows[1]["target_plane_uniformity"]
    assert best["delta_z_mm"] == 0.0
```

- [ ] **Step 2: Run the metric test and verify RED**

Run: `/home/public/zcq/ccbc/slm_code/.venv/bin/python -m pytest tests/test_experimental_delta_z_scan.py::test_common_spots_and_metrics_prefer_sharp_uniform_low_halo_image -v`

Expected: FAIL because common spot detection and metrics are missing.

- [ ] **Step 3: Implement background correction, common peak detection, fixed circular ROIs, and per-image metrics**

Use `scipy.ndimage.maximum_filter` for candidates, greedy distance suppression, and a nearest-neighbor-derived radius when no radius is supplied. Calculate spot sums, min/max uniformity, CV, inside/outside power, whole-image centroid, equivalent half-maximum diameter, 50% encircled-energy radius, and concentration.

- [ ] **Step 4: Implement scan-normalized ranking and saturation exclusion**

```python
quality = np.mean(np.column_stack([
    high_is_good(uniformity), high_is_good(sharpness),
    low_is_good(halo), low_is_good(fwhm),
]), axis=1)
eligible = saturation_fraction <= maximum_saturation_fraction
best = rows[np.argmax(np.where(eligible, quality, -np.inf))]
```

Add a test in which the numerically sharpest point is saturated and verify it is not recommended.

- [ ] **Step 5: Run complete tests and commit the analysis layer**

Run: `/home/public/zcq/ccbc/slm_code/.venv/bin/python -m pytest -q`

Expected: all tests pass.

```bash
git add experimental_delta_z_scan.py tests/test_experimental_delta_z_scan.py
git commit -m "feat: rank measured delta-z image quality"
```

### Task 4: Windows Hardware Adapters, CLI, Plots, and Documentation

**Files:**
- Modify: `experimental_delta_z_scan.py`
- Modify: `requirements-delta-z.txt`
- Create: `README.md`
- Modify: `tests/test_experimental_delta_z_scan.py`

**Interfaces:**
- Consumes: all pure and orchestration interfaces from Tasks 1-3.
- Produces: `SecondaryMonitorSLM(monitor_index: int)` wrapping `slmpy.SLMdisplay`.
- Produces: `main() -> None` CLI that lazily constructs `SecondaryMonitorSLM` and `avt.VimbaCamera`.

- [ ] **Step 1: Write failing CLI/parser and metadata tests**

Assert defaults of monitor 1, 50 us exposure, 16 frames, 1 s settling, correction BMP `CAL_LSH0804730_785nm.bmp`, and LUT 224. Assert `--no-calibration`, ROI, peak distance, spot radius, and raw-frame options parse correctly without importing `wx` or `vmbpy`.

- [ ] **Step 2: Run parser tests and verify RED**

Run: `/home/public/zcq/ccbc/slm_code/.venv/bin/python -m pytest tests/test_experimental_delta_z_scan.py -v`

Expected: parser/default assertions fail until the CLI exists.

- [ ] **Step 3: Implement lazy adapters, complete outputs, and cleanup**

Use `contextlib.ExitStack` so camera and SLM close on success and error. Write experimental CSV/plot, detected-spots overlay, parameters JSON, and best-delta JSON only after successful analysis.

- [ ] **Step 4: Document Windows installation and experiment procedure**

Document Vimba X plus `pip install "vmbpy[numpy]" wxPython`, Windows extended-display mode at 1280 x 1024 and 100% scaling, fixed mechanics, exposure preview, coarse scan, experimental scan, saturation handling, and fine scan.

- [ ] **Step 5: Run offline help and complete tests**

Run: `/home/public/zcq/ccbc/slm_code/.venv/bin/python experimental_delta_z_scan.py --help`

Run: `/home/public/zcq/ccbc/slm_code/.venv/bin/python -m pytest -q`

Expected: help succeeds without hardware packages; all tests pass.

- [ ] **Step 6: Commit the Windows entry point and user guide**

```bash
git add experimental_delta_z_scan.py requirements-delta-z.txt README.md tests/test_experimental_delta_z_scan.py
git commit -m "docs: add automated AVT delta-z experiment workflow"
```

### Task 5: Verification, Review, and GitHub Delivery

**Files:**
- Modify only files required by verification findings.

**Interfaces:**
- Consumes: completed feature and test suite.
- Produces: a verified branch pushed to GitHub without force-push.

- [ ] **Step 1: Run static and full test verification**

```bash
/home/public/zcq/ccbc/slm_code/.venv/bin/python -m py_compile WGS.py delta_z_scan.py experimental_delta_z_scan.py avt.py slmpy.py
/home/public/zcq/ccbc/slm_code/.venv/bin/python -m pytest -q
git diff --check main...HEAD
```

- [ ] **Step 2: Run an in-memory full acquisition/analysis smoke test**

Create three tiny scan BMPs and synthetic camera frames, execute acquisition plus analysis, and verify `experimental_metrics.csv`, plot, spot overlay, parameters, and best-delta JSON exist and agree on the selected `delta_z`.

- [ ] **Step 3: Self-review against the design spec**

Check independent WGS provenance, exact pixel placement, fixed exposure/timing, common ROI use, saturation exclusion, close-on-error, and output traceability.

- [ ] **Step 4: Commit any verification fixes**

```bash
git add -- experimental_delta_z_scan.py avt.py slmpy.py tests/test_experimental_delta_z_scan.py README.md requirements-delta-z.txt
git commit -m "fix: harden hardware delta-z scan verification"
```

- [ ] **Step 5: Push the feature branch using the repository's verified authentication path**

Inspect `origin`, verify available noninteractive authentication, and push `feat/hardware-delta-z-loop` without changing the saved fetch URL or force-pushing.
