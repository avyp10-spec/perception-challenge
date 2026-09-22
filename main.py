"""Part A: archive audit, robust landmark positions, and translation-only BEV."""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
DIAG = ROOT / "diagnostics"
FPS = 30  # Assumed: approximately ten seconds, 299 frames; no timestamps supplied.
EXPECTED = set(range(299))
PATCH_RADIUS = 5


def index_archives():
    """Test every ZIP and index exact member names; never extract or edit data."""
    stores = {"rgb": {}, "xyz": {}}
    report = []
    handles = []
    for path in sorted(DATA.rglob("*.zip")):
        archive = zipfile.ZipFile(path)
        handles.append(archive)
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Corrupt archive: {path.name}, member {bad}")
        ids = {"rgb": [], "xyz": []}
        for member in archive.infolist():
            match = re.search(r"(left|depth)(\d+)\.(png|npz)$", member.filename)
            if not match:
                continue
            kind = "rgb" if match[1] == "left" else "xyz"
            frame = int(match[2])
            if frame in stores[kind]:
                raise ValueError(f"Duplicate {kind} frame {frame}: {path.name}")
            stores[kind][frame] = (archive, member)
            ids[kind].append(frame)
        entry = {"archive": path.name, "bytes": path.stat().st_size,
                 "crc": "passed", "ids": ids}
        report.append(entry)
        print(f"{path.name}: {path.stat().st_size:,} bytes; CRC passed; "
              f"RGB {len(ids['rgb'])}, XYZ {len(ids['xyz'])}", flush=True)
    for kind, frames in stores.items():
        missing, extra = sorted(EXPECTED - frames.keys()), sorted(frames.keys() - EXPECTED)
        print(f"{kind}: {len(frames)} unique; missing={missing}; extra={extra}")
        if missing or extra:
            raise ValueError(f"Incomplete {kind}: missing {missing}, extra {extra}")
    return stores, handles, report


def read_boxes():
    """Interpolate only interior all-zero boxes, retaining original annotations."""
    candidates = [p for p in DATA.rglob("*.csv")
                  if p.name in ("bbox_light.csv", "bboxes_light.csv")]
    if len(candidates) != 1:
        raise ValueError(f"Expected one bounding-box CSV, found {candidates}")
    with candidates[0].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    aliases = (("frame", "x1", "y1", "x2", "y2"),
               ("frame_id", "x_min", "y_min", "x_max", "y_max"))
    columns = next((a for a in aliases if all(k in rows[0] for k in a)), None)
    if columns is None:
        raise ValueError("Unrecognized bounding-box CSV columns")
    ids = [int(row[columns[0]]) for row in rows]
    duplicate = [i for i, count in Counter(ids).items() if count > 1]
    if set(ids) != EXPECTED or duplicate:
        raise ValueError(f"CSV missing={sorted(EXPECTED-set(ids))}; duplicates={duplicate}")
    original = np.empty((299, 4))
    for frame, row in zip(ids, rows):
        original[frame] = [float(row[k]) for k in columns[1:]]
    boxes = original.copy()
    empty = np.all(boxes == 0, axis=1)
    valid = np.flatnonzero(~empty)
    if empty[0] or empty[-1]:
        raise ValueError("Cannot interpolate an empty endpoint bounding box")
    for col in range(4):
        boxes[empty, col] = np.interp(np.flatnonzero(empty), valid, boxes[valid, col])
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Invalid bounding-box coordinates")
    print(f"CSV: 299 unique IDs 0–298; interpolated boxes {np.flatnonzero(empty).tolist()}")
    return original, boxes, empty


def load_array(entry):
    archive, member = entry
    with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as payload:
        if payload.files != ["xyz"]:
            raise ValueError(f"Unexpected NPZ keys in {member.filename}: {payload.files}")
        array = payload["xyz"]
    if array.shape != (1200, 1920, 4) or array.dtype != np.float32:
        raise ValueError(f"Unexpected shape/dtype in {member.filename}: {array.shape}, {array.dtype}")
    return array


def load_rgb(entry):
    archive, member = entry
    with Image.open(io.BytesIO(archive.read(member))) as image:
        image.load()  # Decode every image, not just its ZIP CRC.
        if image.size != (1920, 1200):
            raise ValueError(f"Unexpected RGB size in {member.filename}: {image.size}")
        return np.array(image.convert("RGB"))


def verify_channels(array):
    """Verify perspective geometry on a grid, and fourth-channel values everywhere."""
    sample = array[::20, ::20, :3]
    v, u = np.mgrid[0:1200:20, 0:1920:20]
    valid = np.isfinite(sample).all(axis=2) & (sample[..., 0] > 0.5)
    if valid.sum() < 100:
        raise ValueError("Too few valid pixels to verify channel geometry")
    fits = []
    residuals = []
    for channel, pixels in ((1, u), (2, v)):
        ratio = sample[..., channel][valid] / sample[..., 0][valid]
        fit = np.polyfit(pixels[valid], ratio, 1)
        residual = float(np.median(np.abs(ratio - np.polyval(fit, pixels[valid]))))
        if fit[0] >= 0 or residual > 1e-4:
            raise ValueError("Channel axes differ from verified X-forward, Y-left, Z-up geometry")
        fits.append(fit.tolist())
        residuals.append(residual)
    xyz_valid = np.isfinite(array[..., :3]).all(axis=2)
    fourth = array[..., 3]
    if not np.array_equal(np.isfinite(fourth), xyz_valid):
        raise ValueError("Fourth-channel validity differs from XYZ")
    if np.any(fourth[xyz_valid] != 0):
        raise ValueError("Fourth channel has nonzero values: inspect before processing")
    return {"Y_over_X_vs_u": fits[0], "Z_over_X_vs_v": fits[1],
            "median_residuals": residuals}


def estimate_position(array, box):
    """Use an 11x11 center patch, clipped to the box, and a robust depth gate."""
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1200):
        raise ValueError("Bounding box extends outside image")
    u, v = np.rint([(x1+x2)/2, (y1+y2)/2]).astype(int)
    left, right = max(int(np.ceil(x1)), u-PATCH_RADIUS), min(int(np.ceil(x2)), u+PATCH_RADIUS+1)
    top, bottom = max(int(np.ceil(y1)), v-PATCH_RADIUS), min(int(np.ceil(y2)), v+PATCH_RADIUS+1)
    points = array[top:bottom, left:right, :3].reshape(-1, 3)
    mask = (np.isfinite(points).all(axis=1) & (points[:, 0] > 0.5)
            & (np.linalg.norm(points, axis=1) < 60)
            & (np.abs(points[:, 1]) < 40) & (np.abs(points[:, 2]) < 15))
    points = points[mask]
    info = {"u": int(u), "v": int(v), "left": left, "right": right,
            "top": top, "bottom": bottom, "valid_points": len(points), "used_points": 0}
    if len(points) < 5:
        return np.full(3, np.nan), info
    depth = points[:, 0]
    median = np.median(depth)
    mad = np.median(np.abs(depth - median))
    points = points[np.abs(depth-median) <= max(0.30, 3 * 1.4826 * mad)]
    info["used_points"] = len(points)
    if len(points) < 5:
        return np.full(3, np.nan), info
    return np.median(points, axis=0), info


def rolling_median(values, radius):
    result = values.copy()
    for i in range(len(values)):
        window = values[max(0, i-radius):min(len(values), i+radius+1)]
        good = np.isfinite(window).all(axis=1)
        if good.any():
            result[i] = np.median(window[good], axis=0)
    return result


def filter_positions(raw):
    """Reject isolated large local deviations; preserve gaps and the frame-0 anchor."""
    local = rolling_median(raw, 3)
    distance = np.linalg.norm(raw-local, axis=1)
    rejected = np.isfinite(raw).all(axis=1) & (distance > 1.0)
    clean = raw.copy()
    clean[rejected] = np.nan
    filtered = rolling_median(clean, 2)
    filtered[~np.isfinite(clean).all(axis=1)] = np.nan
    # Three-frame weighted mean softens median-filter steps, without bridging gaps.
    softened = filtered.copy()
    for i in range(1, len(filtered)-1):
        window = filtered[i-1:i+2]
        if np.isfinite(window).all():
            softened[i] = np.average(window, axis=0, weights=[1, 2, 1])
    filtered = softened
    # Keep true endpoint measurements; never substitute a later t=0.
    filtered[0] = clean[0]
    filtered[-1] = clean[-1]
    if not np.isfinite(filtered[0]).all():
        raise ValueError("Frame 0 has no trustworthy position; cannot establish t=0")
    return filtered, rejected


def to_world(points, anchor):
    """Actual channels: forward, LEFT, up. Rotate the frame-0 ray onto world +X."""
    angle = np.arctan2(anchor[1], anchor[0])
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, s], [-s, c]])
    return -(points[:, :2] @ rotation.T)


def save_overlay(rgb, box, info, frame, interpolated):
    fig, ax = plt.subplots(figsize=(10, 5))
    x1, y1, x2, y2 = box
    ax.imshow(rgb)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="lime", linewidth=2))
    ax.add_patch(Rectangle((info["left"], info["top"]),
                          info["right"]-info["left"], info["bottom"]-info["top"],
                          fill=False, edgecolor="orange", linewidth=2))
    ax.plot(info["u"], info["v"], "+", color="red", markersize=8)
    ax.set_title(f"Frame {frame}: green box {'(interpolated)' if interpolated else '(provided)'}; orange sample patch")
    ax.axis("off")
    fig.tight_layout()
    name = "rgb_bbox_patch" if frame == 0 else f"rgb_bbox_patch_{frame:03d}"
    fig.savefig(DIAG / f"{name}.png", dpi=140)
    ax.set_xlim(max(0, x1-45), min(1920, x2+45))
    ax.set_ylim(min(1200, y2+45), max(0, y1-45))
    fig.savefig(DIAG / f"{name}_zoom.png", dpi=140)
    plt.close(fig)


def draw_outputs(raw_world, world):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(0, 0, marker="*", color="gold", edgecolors="black", s=150, label="Traffic light / ground origin")
    ax.plot(raw_world[:, 0], raw_world[:, 1], ".", color="silver", markersize=4, label="Raw")
    ax.plot(world[:, 0], world[:, 1], color="steelblue", linewidth=1, label="Filtered")
    ax.scatter(world[0, 0], world[0, 1], color="green", label="Frame 0", zorder=5)
    ax.scatter(world[-1, 0], world[-1, 1], color="crimson", label="Frame 298", zorder=5)
    ax.set(xlabel="World X (m): initial car-to-light direction",
           ylabel="World Y (m): left", title="Translation-only ego trajectory; fixed initial heading")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.18), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "trajectory.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    all_points = np.vstack([raw_world, world, [0, 0]])
    lo, hi = np.nanmin(all_points, axis=0)-2, np.nanmax(all_points, axis=0)+2
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]),
           xlabel="World X (m)", ylabel="World Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.scatter(0, 0, marker="*", s=150, color="gold", edgecolors="black", label="Traffic light")
    raw_line, = ax.plot([], [], ".", color="silver", markersize=4, label="Raw")
    line, = ax.plot([], [], "-", color="steelblue", linewidth=1.5, label="Filtered")
    car, = ax.plot([], [], "o", color="crimson")
    ax.legend()
    fig.tight_layout()
    writer = cv2.VideoWriter(str(ROOT / "trajectory.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (900, 600))
    if not writer.isOpened():
        raise RuntimeError("MP4 encoder unavailable")
    try:
        for i in range(299):
            raw_line.set_data(raw_world[:i+1, 0], raw_world[:i+1, 1])
            line.set_data(world[:i+1, 0], world[:i+1, 1])
            car.set_data([world[i, 0]], [world[i, 1]])
            state = "" if np.isfinite(world[i]).all() else " — unavailable"
            ax.set_title(f"Frame {i}/298 | t={i/FPS:.2f}s (30 fps assumed){state}")
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        plt.close(fig)


def validate_outputs():
    for path in [ROOT / "trajectory.png", *DIAG.glob("rgb_bbox_patch*.png")]:
        with Image.open(path) as image:
            image.load()
            if image.width == 0 or image.height == 0:
                raise ValueError(f"Invalid PNG: {path}")
    capture = cv2.VideoCapture(str(ROOT / "trajectory.mp4"))
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape != (600, 900, 3):
            raise ValueError("Unexpected video dimensions")
        count += 1
    capture.release()
    if count != 299:
        raise ValueError(f"MP4 decoded only {count}/299 frames")
    print("Validated PNGs and decoded all 299 MP4 frames.")
    return count


def main():
    stores, handles, archive_report = index_archives()
    try:
        original, boxes, interpolated = read_boxes()
        raw = np.full((299, 3), np.nan)
        metadata, geometry = [], []
        previews = {}
        for i in range(299):
            rgb = load_rgb(stores["rgb"][i])
            array = load_array(stores["xyz"][i])
            geometry.append(verify_channels(array))
            raw[i], info = estimate_position(array, boxes[i])
            metadata.append(info)
            if i in (0, 3, 4, 5, 23, 150, 298):
                previews[i] = rgb
            if i % 50 == 0:
                print(f"Decoded RGB/XYZ and checked channels: {i+1}/299", flush=True)
        filtered, rejected = filter_positions(raw)
        world = to_world(filtered, filtered[0])
        raw_world = to_world(raw, filtered[0])
        valid = np.isfinite(filtered).all(axis=1)
        steps = np.linalg.norm(np.diff(world, axis=0), axis=1)
        raw_steps = np.linalg.norm(np.diff(raw_world, axis=0), axis=1)
        largest = np.argsort(np.nan_to_num(steps, nan=-1))[-10:][::-1]
        summary = {
            "rgb_frames": 299, "xyz_frames": 299, "csv_rows": 299,
            "interpolated_boxes": np.flatnonzero(interpolated).tolist(),
            "invalid_depth_frames": np.flatnonzero(~np.isfinite(raw).all(axis=1)).tolist(),
            "outlier_frames": np.flatnonzero(rejected).tolist(),
            "usable_frames": int(valid.sum()), "t0_frame": 0,
            "world_x_range_m": [float(np.nanmin(world[:, 0])), float(np.nanmax(world[:, 0]))],
            "world_y_range_m": [float(np.nanmin(world[:, 1])), float(np.nanmax(world[:, 1]))],
            "median_step_m": float(np.nanmedian(steps)), "max_step_m": float(np.nanmax(steps)),
            "raw_max_step_m": float(np.nanmax(raw_steps)),
            "largest_steps": [{"from": int(i), "to": int(i+1), "meters": float(steps[i])} for i in largest],
            "channel_geometry_frame0": geometry[0],
            "fourth_channel": "All 299 arrays: zero at finite XYZ points; nonfinite elsewhere. Unused padding inferred.",
            "archives": archive_report,
        }
        print(json.dumps({k:v for k,v in summary.items() if k != "archives"}, indent=2), flush=True)
        DIAG.mkdir(exist_ok=True)
        rows = []
        for i in range(299):
            row = {"frame_id": i, "time_s": i/FPS, "bbox_interpolated": bool(interpolated[i]),
                   "status": "outlier" if rejected[i] else ("ok" if valid[i] else "invalid_depth"),
                   **metadata[i]}
            for name, data in (("original", original), ("bbox", boxes)):
                row.update({f"{name}_{k}": float(data[i,j]) for j,k in enumerate(("x1","y1","x2","y2"))})
            for name, data, axes in (("raw", raw, "XYZ"), ("filtered", filtered, "XYZ"),
                                      ("raw_ego", raw_world, "xy"), ("ego", world, "xy")):
                row.update({f"{name}_{axis}_m": float(data[i,j]) if np.isfinite(data[i,j]) else ""
                            for j,axis in enumerate(axes)})
            rows.append(row)
        with (DIAG / "traffic_light_positions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for i, rgb in previews.items():
            save_overlay(rgb, boxes[i], metadata[i], i, interpolated[i])
        draw_outputs(raw_world, world)
        summary["video_frames"] = validate_outputs()
        summary["output_bytes"] = {str(p.relative_to(ROOT)): p.stat().st_size
                                   for p in [ROOT / "trajectory.png", ROOT / "trajectory.mp4",
                                             DIAG / "traffic_light_positions.csv"]}
        (DIAG / "validation.json").write_text(json.dumps(summary, indent=2)+"\n")
        write_readme(summary)
        print("Output bytes:", summary["output_bytes"])
    finally:
        for archive in handles:
            archive.close()


def write_readme(summary):
    """Keep the short report synchronized with the verified run."""
    x = summary["world_x_range_m"]
    y = summary["world_y_range_m"]
    text = f"""# Wisconsin Autonomous — Part A

![Ego trajectory with raw and filtered positions](trajectory.png)

[Watch or download the trajectory video](trajectory.mp4)

**Data.** All three ZIPs pass CRC checks: 299 RGB images, 299 unique XYZ arrays,
and 299 CSV rows, IDs 0–298; no missing or duplicate IDs. Archives are read in
place. Actual names are `rgb/leftNNNNNN.png`, `xyz/depthNNNNNN.npz`, and
`bbox_light.csv` (columns `frame,x1,y1,x2,y2`), unlike the challenge examples.
XYZ is float32 `(1200,1920,4)` under key `xyz`. Use both matching
`xyz-20260921T212945Z-1-001.zip` and `xyz-20260921T212945Z-1-002.zip`.

**Channel verification.** Across all 299 arrays, perspective ratios satisfy
Y/X ≈ −0.000783813u + 0.733322 and Z/X ≈ −0.000783813v + 0.473695;
frame-0 median residuals are below 1e−8. This identifies channels 0–2 as
forward, **left**, up: positive Y points toward image-left, contrary to the
challenge's “Y right” text. Metric scale follows the supplied meters definition.
Channel 3 is zero at every finite XYZ point and nonfinite elsewhere in every
array. It appears to be unused padding, not color/confidence; its producer
semantics cannot be proven from these files.

**Method.** Linearly interpolate each box coordinate for frames 3–5 between 2
and 6, and frame 23 between 22 and 24, without editing the CSV. Sample an 11×11
center patch clipped to the box. Reject nonfinite, zero/behind-camera, and
implausible points (X > 0.5 m, range < 60 m, |Y| < 40 m, |Z| < 15 m).
Reject depth deviations beyond max(0.30 m, 3×1.4826×MAD); take the coordinate
median of at least five remaining points. Remove positions >1 m from a
seven-frame local median, then apply a five-frame median and three-frame
weighted mean [1,2,1] within valid stretches. Preserve gaps and frame-0/end
measurements. Raw positions and every rejection remain visible in CSV/plot.

**World frame and assumptions.** Origin is under the light; +Z up. Let
θ=atan2(Y₀,X₀) from actual frame 0. Ego XY = −R(−θ)[X,Y], aligning the
initial car-to-light ray with world +X and world +Y left. Assume fixed camera
orientation (yaw, pitch, roll) and mounting; one landmark cannot recover both
translation and changing orientation. Thus lateral curvature may include
turning effects, not true lateral travel. Camera height is irrelevant to XY.
Time uses assumed 30 fps (no timestamps), t=frame_id/30, with t=0 at frame 0.

**Results.** {summary["usable_frames"]}/299 filtered positions; no failed depth patches,
{len(summary["outlier_frames"])} rejected outliers. X: {x[0]:.2f} to {x[1]:.2f} m;
Y: {y[0]:.2f} to {y[1]:.2f} m. Median/max adjacent valid step:
{summary["median_step_m"]:.3f}/{summary["max_step_m"]:.3f} m (raw max
{summary["raw_max_step_m"]:.3f} m). Early long-range stereo jitter remains;
this is not an independently validated odometry estimate. PNGs open and all
299 MP4 frames decode, including unavailable-position slots (9.97 s).

**Reproduce (PowerShell, project root):**
```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe main.py
```
Outputs: `trajectory.png`, `trajectory.mp4`, and `diagnostics/` (raw/filtered
CSV, box/patch overlays and crops, validation JSON). Dataset, ZIPs, environment
and caches are gitignored; outputs are retained.
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
