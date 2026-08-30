# Hardware Delta-Z Scan Design

## Goal

Extend the existing independently optimized pupil-conjugation `delta_z` scan
into a repeatable experimental loop:

1. display each generated phase pattern on the Hamamatsu SLM through the
   Windows secondary-monitor path already used by the notebooks;
2. wait for the optical response to settle;
3. acquire and average frames from an Allied Vision camera through Vimba X;
4. measure the real target-plane spot quality; and
5. save traceable data and recommend the best coarse-scan `delta_z` for a
   later fine scan.

The objective relay and CCD remain mechanically fixed for the entire scan.

## Confirmed Hardware Path

- The SLM is driven as Windows monitor 1 through `slmpy.SLMdisplay`, matching
  `test tweezer generation vimba.ipynb`.
- The camera is Allied Vision/AVT and is acquired through the existing
  `avt.VimbaCamera` wrapper and Vimba X Python package (`vmbpy`).
- The WGS phase for every `delta_z` is generated independently by the existing
  `delta_z_scan.py`; the hardware loop consumes its ordered BMP manifest.
- The generated BMP is the raw 1272 x 1024 LCOS phase. Factory correction and
  phase-level LUT conversion are applied only at display time and recorded in
  the experimental metadata.

## Architecture

`delta_z_scan.py` remains the optical-model/offline generator. A new
`experimental_delta_z_scan.py` module owns the hardware experiment. Keeping
the stages separate makes it possible to repeat camera acquisition without
rerunning a costly 4096 x 4096 WGS scan and prevents a camera failure from
leaving a partially generated optical scan.

The experimental module has four boundaries:

- scan manifest loading: read `delta_z_metrics.csv`, resolve each BMP, retain
  its `scan_NNN` order, and reject missing/duplicate artifacts;
- SLM preparation: optionally add the matching factory correction and LUT,
  then center-pad 1272 x 1024 to the monitor transport resolution (normally
  1280 x 1024) without interpolation;
- acquisition: show a frame, wait, capture a fixed number of equally exposed
  AVT images, and save their floating-point mean plus an optional raw stack;
- analysis: find the expected number of spots once from a normalized
  across-scan reference image, reuse those fixed ROIs for every `delta_z`, and
  calculate comparable experimental metrics.

Hardware objects are injected into the acquisition function. Tests use simple
in-memory display and camera doubles, while the CLI lazily imports `wxPython`
and `vmbpy` only when real hardware is requested.

## Exact Pixel Handling

The phase BMP must never be resized. For a 1280 x 1024 display transport, the
1272 x 1024 LCOS phase is copied into columns 4 through 1275 and the remaining
four columns on each side are filled with phase level zero. Any display smaller
than the phase image is rejected. The experiment records both the active LCOS
shape and display transport shape.

The default calibration follows the working AVT notebook:

- correction: `CAL_LSH0804730_785nm.bmp` (the closest supplied calibration to
  the 795 nm operating wavelength);
- LUT level: 224.

Both are CLI-configurable and `--no-calibration` preserves the raw BMP. The
selected correction filename and LUT are recorded so scans can be compared.

## Acquisition Behavior

For each ordered scan point:

1. prepare and display the phase image;
2. wait `settle_seconds` (default 1.0 s);
3. capture `frames_per_point` frames at one fixed exposure (default 16 frames);
4. verify that all frame shapes match;
5. calculate the floating-point mean;
6. record the maximum raw value and saturated-pixel fraction;
7. save `camera_average_<scan label>.npy` and a viewable 16-bit TIFF;
8. optionally save `camera_raw_<scan label>.npy`.

The SLM frame remains displayed if a camera exception occurs, but both camera
and SLM handles are always closed by the CLI. Existing outputs are never
silently overwritten; acquisition requires a fresh experiment directory.

## Experimental Image Metrics

All averaged images are background-corrected by subtracting a configurable
low percentile and clipping negative values. An optional camera ROI restricts
spot detection. The expected spot count comes from `scan_parameters.json`
(`target_array_size`, normally 8 x 8) unless explicitly overridden.

Peak positions are detected once from the mean of total-power-normalized scan
images. A non-maximum-distance rule selects the expected number of stable
spots. A circular spot radius is either supplied or inferred from nearest
neighbor spacing. The same centers and radius are used for every scan point.

The CSV contains:

- `delta_z_mm` and source BMP;
- measured target uniformity (`min spot sum / max spot sum`);
- spot coefficient of variation (`std / mean`);
- target efficiency and background/halo fraction;
- peak intensity and saturated-pixel fraction;
- whole-pattern centroid, global peak location, and their offset;
- mean equivalent FWHM diameter;
- mean 50% encircled-energy radius;
- mean spot concentration/sharpness;
- a normalized `quality_score`.

The quality score is the equal-weight mean of four scan-normalized components:
higher uniformity, higher spot sharpness, lower halo, and lower equivalent
FWHM. It is a ranking aid, not a replacement for inspecting the raw metrics.
Saturated scan points are marked and excluded from automatic recommendation.
If all points are saturated or spot detection fails, the program stops without
claiming a best `delta_z`.

## Outputs

The experiment directory contains:

- averaged CCD arrays and TIFF images for each `delta_z`;
- optional raw frame stacks;
- `experimental_metrics.csv`;
- `experimental_metrics_vs_delta_z.png`;
- `detected_spots.png` showing the common analysis ROIs;
- `best_delta_z.json` with the selected coarse point and score;
- `experimental_parameters.json` containing camera, display, calibration,
  timing, ROI, and source-scan provenance.

## User Workflow

Generate the independently optimized scan first:

```powershell
python delta_z_scan.py --output-dir delta_z_scan_outputs
```

Then connect the SLM as the second Windows display, connect the AVT camera, and
run the experimental loop:

```powershell
python experimental_delta_z_scan.py `
  --scan-dir delta_z_scan_outputs `
  --output-dir delta_z_experiment `
  --monitor 1 `
  --exposure-us 50 `
  --frames-per-point 16 `
  --settle-seconds 1
```

After identifying the best coarse point, generate a fresh fine-scan directory
with `--scan-start-mm`, `--scan-stop-mm`, and `--scan-step-mm`, then repeat the
same acquisition command.

