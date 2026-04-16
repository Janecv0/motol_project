#!/usr/bin/env python3
"""Standalone experiment CSV visualizer for tension and blob movement."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

NUM_BLOBS = 5
RUN_PREFIXES = ("experiment_", "experiment2_")
DATA_SUFFIX = "_data.csv"
SETUP_SUFFIX = "_setup.csv"
TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")
DEFAULT_PLAYBACK_DT_S = 0.1

BLOB_COLORS = {
    0: "#0072B2",  # blue
    1: "#E69F00",  # orange
    2: "#009E73",  # green
    3: "#D55E00",  # vermillion
    4: "#CC79A7",  # magenta
}


@dataclass
class ExperimentEntry:
    run_id: str
    data_path: Path
    setup_path: Optional[Path]
    sort_key: datetime


@dataclass
class ExperimentData:
    run_id: str
    data_path: Path
    setup_path: Optional[Path]
    time_values: np.ndarray
    time_mode: str
    time_label: str
    tension_a: np.ndarray
    tension_b: np.ndarray
    blob_x: Dict[int, np.ndarray]
    blob_y: Dict[int, np.ndarray]
    phase: List[str]
    repetition: np.ndarray
    pulses_moved: np.ndarray
    setup_metadata: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def num_rows(self) -> int:
        return int(self.time_values.size)


def _parse_float(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _parse_timestamp(value: object) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _parse_run_stamp(run_id: str) -> Optional[datetime]:
    for prefix in RUN_PREFIXES:
        if not run_id.startswith(prefix):
            continue
        stamp = run_id[len(prefix) :]
        try:
            return datetime.strptime(stamp, "%Y%m%d_%H%M%S")
        except ValueError:
            return None
    return None


def discover_experiments(experiments_dir: Path) -> List[ExperimentEntry]:
    if not experiments_dir.is_dir():
        return []

    entries: List[ExperimentEntry] = []
    seen_paths: set[Path] = set()
    for prefix in RUN_PREFIXES:
        for data_path in experiments_dir.glob(f"{prefix}*{DATA_SUFFIX}"):
            if not data_path.is_file() or data_path in seen_paths:
                continue
            seen_paths.add(data_path)

            run_id = data_path.name[: -len(DATA_SUFFIX)]
            setup_candidate = data_path.with_name(f"{run_id}{SETUP_SUFFIX}")
            setup_path = setup_candidate if setup_candidate.is_file() else None
            sort_key = _parse_run_stamp(run_id)
            if sort_key is None:
                sort_key = datetime.fromtimestamp(data_path.stat().st_mtime)

            entries.append(
                ExperimentEntry(
                    run_id=run_id,
                    data_path=data_path,
                    setup_path=setup_path,
                    sort_key=sort_key,
                )
            )

    entries.sort(key=lambda item: item.sort_key, reverse=True)
    return entries


def _read_setup_metadata(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.is_file():
        return {}

    metadata: Dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            metadata[row[0]] = row[1]

    return metadata


def load_experiment(entry: ExperimentEntry) -> ExperimentData:
    warnings: List[str] = []
    with entry.data_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {entry.data_path.name}")

    for required in ("tension_a_g", "tension_b_g"):
        if required not in fieldnames:
            warnings.append(f"Missing '{required}' column.")
    if "timestamp_local" not in fieldnames:
        warnings.append("Missing 'timestamp_local' column.")

    for blob_idx in range(NUM_BLOBS):
        for axis in ("x", "y"):
            col = f"blob{blob_idx}_{axis}"
            if col not in fieldnames:
                warnings.append(f"Missing '{col}' column.")

    size = len(rows)
    tension_a = np.full(size, np.nan, dtype=float)
    tension_b = np.full(size, np.nan, dtype=float)
    repetition = np.full(size, np.nan, dtype=float)
    pulses_moved = np.full(size, np.nan, dtype=float)
    phase: List[str] = []
    parsed_timestamps: List[Optional[datetime]] = []
    blob_x = {i: np.full(size, np.nan, dtype=float) for i in range(NUM_BLOBS)}
    blob_y = {i: np.full(size, np.nan, dtype=float) for i in range(NUM_BLOBS)}

    for idx, row in enumerate(rows):
        tension_a[idx] = _parse_float(row.get("tension_a_g"))
        tension_b[idx] = _parse_float(row.get("tension_b_g"))
        repetition[idx] = _parse_float(row.get("repetition"))
        pulses_moved[idx] = _parse_float(row.get("pulses_moved"))
        phase.append((row.get("phase") or "").strip())
        parsed_timestamps.append(_parse_timestamp(row.get("timestamp_local")))

        for blob_idx in range(NUM_BLOBS):
            blob_x[blob_idx][idx] = _parse_float(row.get(f"blob{blob_idx}_x"))
            blob_y[blob_idx][idx] = _parse_float(row.get(f"blob{blob_idx}_y"))

    if all(ts is not None for ts in parsed_timestamps):
        first_ts = parsed_timestamps[0]
        assert first_ts is not None  # for type checkers
        time_values = np.array(
            [(ts - first_ts).total_seconds() for ts in parsed_timestamps], dtype=float
        )
        time_mode = "elapsed_s"
        time_label = "Elapsed time [s]"
    else:
        time_values = np.arange(size, dtype=float)
        time_mode = "sample_index"
        time_label = "Sample index"
        warnings.append("Malformed or missing timestamps detected. Using sample index.")

    setup_metadata = _read_setup_metadata(entry.setup_path)

    return ExperimentData(
        run_id=entry.run_id,
        data_path=entry.data_path,
        setup_path=entry.setup_path,
        time_values=time_values,
        time_mode=time_mode,
        time_label=time_label,
        tension_a=tension_a,
        tension_b=tension_b,
        blob_x=blob_x,
        blob_y=blob_y,
        phase=phase,
        repetition=repetition,
        pulses_moved=pulses_moved,
        setup_metadata=setup_metadata,
        warnings=warnings,
    )


class ExperimentVisualizerApp:
    def __init__(self, root: tk.Tk, experiments_dir: Path, initial_experiment: Optional[str]) -> None:
        self.root = root
        self.experiments_dir = experiments_dir
        self.initial_experiment = initial_experiment

        self.entries: List[ExperimentEntry] = []
        self.entries_by_run_id: Dict[str, ExperimentEntry] = {}
        self.current_data: Optional[ExperimentData] = None
        self.current_frame = 0
        self.playing = False
        self.playback_after_id: Optional[str] = None
        self.updating_slider = False

        self.speed_values = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}

        self.experiment_var = tk.StringVar()
        self.blob_var = tk.StringVar(value="blob0")
        self.speed_var = tk.StringVar(value="1x")
        self.status_var = tk.StringVar(value="Ready")
        self.frame_var = tk.StringVar(value="Frame: 0/0")

        self._build_ui()
        self.refresh_experiments(preferred_run_id=initial_experiment)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        self.root.title("Experiment Visualizer")
        self.root.geometry("1450x880")

        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill="x")

        ttk.Label(controls, text="Experiment:").grid(row=0, column=0, padx=(0, 6), sticky="w")
        self.experiment_combo = ttk.Combobox(
            controls,
            textvariable=self.experiment_var,
            state="readonly",
            width=42,
        )
        self.experiment_combo.grid(row=0, column=1, columnspan=3, sticky="ew")
        self.experiment_combo.bind("<<ComboboxSelected>>", self.on_experiment_selected)

        self.refresh_button = ttk.Button(controls, text="Refresh", command=self.on_refresh_clicked)
        self.refresh_button.grid(row=0, column=4, padx=6)

        self.play_button = ttk.Button(controls, text="Play", command=self.toggle_playback, state="disabled")
        self.play_button.grid(row=0, column=5, padx=6)

        ttk.Label(controls, text="Speed:").grid(row=0, column=6, padx=(10, 4), sticky="e")
        self.speed_combo = ttk.Combobox(
            controls,
            textvariable=self.speed_var,
            values=list(self.speed_values.keys()),
            state="disabled",
            width=6,
        )
        self.speed_combo.grid(row=0, column=7, sticky="w")

        ttk.Label(controls, text="Blob:").grid(row=0, column=8, padx=(10, 4), sticky="e")
        self.blob_combo = ttk.Combobox(
            controls,
            textvariable=self.blob_var,
            values=[f"blob{i}" for i in range(NUM_BLOBS)],
            state="disabled",
            width=8,
        )
        self.blob_combo.grid(row=0, column=9, sticky="w")
        self.blob_combo.bind("<<ComboboxSelected>>", self.on_blob_changed)

        self.frame_slider = tk.Scale(
            controls,
            from_=0,
            to=0,
            orient="horizontal",
            resolution=1,
            showvalue=False,
            command=self.on_slider_changed,
            state="disabled",
        )
        self.frame_slider.grid(row=1, column=0, columnspan=10, pady=(8, 2), sticky="ew")

        ttk.Label(controls, textvariable=self.frame_var).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Label(controls, textvariable=self.status_var).grid(row=2, column=2, columnspan=8, sticky="e")

        controls.columnconfigure(1, weight=1)

        plots = ttk.Panedwindow(self.root, orient="horizontal")
        plots.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        time_frame = ttk.Frame(plots)
        xy_frame = ttk.Frame(plots)
        plots.add(time_frame, weight=3)
        plots.add(xy_frame, weight=2)

        self.time_fig = Figure(figsize=(8, 6), dpi=100)
        self.ax_tension = self.time_fig.add_subplot(211)
        self.ax_blob_ts = self.time_fig.add_subplot(212, sharex=self.ax_tension)

        self.ax_tension.set_title("Tension over time")
        self.ax_tension.set_ylabel("Tension [g]")
        self.ax_tension.grid(True, alpha=0.3)
        self.line_tension_a, = self.ax_tension.plot([], [], color="#2E86AB", linewidth=1.8, label="tension_a_g")
        self.line_tension_b, = self.ax_tension.plot([], [], color="#F18F01", linewidth=1.8, label="tension_b_g")
        self.cursor_tension = self.ax_tension.axvline(
            x=0.0, color=BLOB_COLORS[0], linestyle=":", linewidth=1.3, label="current frame"
        )
        self.ax_tension.legend(loc="upper right")

        self.ax_blob_ts.set_title("Selected blob coordinates over time")
        self.ax_blob_ts.set_xlabel("Elapsed time [s]")
        self.ax_blob_ts.set_ylabel("Pixels")
        self.ax_blob_ts.grid(True, alpha=0.3)
        self.line_blob_x, = self.ax_blob_ts.plot([], [], color=BLOB_COLORS[0], linewidth=1.8, label="blob0_x")
        self.line_blob_y, = self.ax_blob_ts.plot(
            [],
            [],
            color=BLOB_COLORS[0],
            linewidth=1.8,
            linestyle="--",
            label="blob0_y",
        )
        self.cursor_blob = self.ax_blob_ts.axvline(x=0.0, color=BLOB_COLORS[0], linestyle=":", linewidth=1.3)
        self.ax_blob_ts.legend(loc="upper right")

        self.time_fig.tight_layout(pad=1.4)
        self.time_canvas = FigureCanvasTkAgg(self.time_fig, master=time_frame)
        self.time_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.xy_fig = Figure(figsize=(6, 6), dpi=100)
        self.ax_xy = self.xy_fig.add_subplot(111)
        self.ax_xy.set_title("Blob movement (XY)")
        self.ax_xy.set_xlabel("X [px]")
        self.ax_xy.set_ylabel("Y [px]")
        self.ax_xy.grid(True, alpha=0.3)

        self.blob_trails = {}
        self.blob_markers = {}
        for blob_idx in range(NUM_BLOBS):
            color = BLOB_COLORS[blob_idx]
            trail, = self.ax_xy.plot([], [], color=color, linewidth=1.6, alpha=0.45, label=f"blob{blob_idx}")
            marker, = self.ax_xy.plot(
                [],
                [],
                marker="o",
                linestyle="None",
                markersize=8,
                color=color,
                label=f"_blob{blob_idx}_marker",
            )
            self.blob_trails[blob_idx] = trail
            self.blob_markers[blob_idx] = marker

        legend_handles = [
            Line2D([0], [0], color=BLOB_COLORS[idx], marker="o", linestyle="-", label=f"blob{idx}")
            for idx in range(NUM_BLOBS)
        ]
        self.ax_xy.legend(handles=legend_handles, title="Blobs", loc="best")
        self.ax_xy.set_aspect("equal", adjustable="box")

        self.xy_fig.tight_layout(pad=1.0)
        self.xy_canvas = FigureCanvasTkAgg(self.xy_fig, master=xy_frame)
        self.xy_canvas.get_tk_widget().pack(fill="both", expand=True)

        info_frame = ttk.LabelFrame(self.root, text="Run Info", padding=8)
        info_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.info_text = tk.Text(info_frame, height=8, wrap="word")
        self.info_text.pack(fill="x")
        self.info_text.configure(state="disabled")

    def on_refresh_clicked(self) -> None:
        preferred = self.experiment_var.get().strip() or None
        self.refresh_experiments(preferred_run_id=preferred)

    def refresh_experiments(self, preferred_run_id: Optional[str] = None) -> None:
        self.stop_playback()

        self.entries = discover_experiments(self.experiments_dir)
        self.entries_by_run_id = {entry.run_id: entry for entry in self.entries}
        run_ids = [entry.run_id for entry in self.entries]

        self.experiment_combo["values"] = run_ids
        if not run_ids:
            self.experiment_var.set("")
            self.current_data = None
            self._set_data_controls_enabled(False)
            self._clear_plots()
            self._set_info_text(
                "No files matching "
                f"'{RUN_PREFIXES[0]}*{DATA_SUFFIX}' or "
                f"'{RUN_PREFIXES[1]}*{DATA_SUFFIX}' in:\n"
                f"{self.experiments_dir.resolve()}"
            )
            self.status_var.set("No experiment CSV files found.")
            return

        if preferred_run_id in self.entries_by_run_id:
            selected = preferred_run_id
        else:
            selected = run_ids[0]
        self.experiment_var.set(selected)
        self._load_current_selection()

    def on_experiment_selected(self, event: object = None) -> None:
        self._load_current_selection()

    def _load_current_selection(self) -> None:
        run_id = self.experiment_var.get().strip()
        if not run_id:
            return

        entry = self.entries_by_run_id.get(run_id)
        if entry is None:
            self.status_var.set(f"Unknown experiment selection: {run_id}")
            return

        self.stop_playback()

        try:
            data = load_experiment(entry)
        except Exception as exc:
            self.current_data = None
            self._set_data_controls_enabled(False)
            self._clear_plots()
            self._set_info_text(f"Failed to load {entry.data_path.name}\n\n{exc}")
            self.status_var.set(f"Failed to load {entry.data_path.name}")
            return

        self.current_data = data
        self.current_frame = 0
        self._set_data_controls_enabled(True)

        self.updating_slider = True
        self.frame_slider.configure(from_=0, to=max(data.num_rows - 1, 0))
        self.frame_slider.set(0)
        self.updating_slider = False

        self._update_time_series_static()
        self._configure_xy_axes()
        self._update_frame(0, sync_slider=True)
        self._update_info_panel()

        if data.warnings:
            self.status_var.set(f"{run_id} loaded with warnings.")
        else:
            self.status_var.set(f"{run_id} loaded.")

    def _update_info_panel(self) -> None:
        data = self.current_data
        if data is None:
            self._set_info_text("No run loaded.")
            return

        lines = [
            f"Run ID: {data.run_id}",
            f"Rows: {data.num_rows}",
            f"Data file: {data.data_path}",
            f"Time axis mode: {data.time_mode}",
        ]
        if data.setup_path is not None:
            lines.append(f"Setup file: {data.setup_path}")

        if data.setup_metadata:
            lines.append("")
            lines.append("Setup metadata:")
            for key in sorted(data.setup_metadata.keys()):
                lines.append(f"  {key}: {data.setup_metadata[key]}")

        if data.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in data.warnings:
                lines.append(f"  - {warning}")

        self._set_info_text("\n".join(lines))

    def _set_info_text(self, text: str) -> None:
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.configure(state="disabled")

    def _set_data_controls_enabled(self, enabled: bool) -> None:
        if enabled:
            self.play_button.configure(state="normal")
            self.frame_slider.configure(state="normal")
            self.speed_combo.configure(state="readonly")
            self.blob_combo.configure(state="readonly")
        else:
            self.play_button.configure(state="disabled", text="Play")
            self.frame_slider.configure(state="disabled")
            self.speed_combo.configure(state="disabled")
            self.blob_combo.configure(state="disabled")
            self.frame_var.set("Frame: 0/0")

    def _clear_plots(self) -> None:
        self.line_tension_a.set_data([], [])
        self.line_tension_b.set_data([], [])
        self.line_blob_x.set_data([], [])
        self.line_blob_y.set_data([], [])
        self.cursor_tension.set_xdata([0.0, 0.0])
        self.cursor_blob.set_xdata([0.0, 0.0])

        self.ax_tension.set_xlim(0.0, 1.0)
        self.ax_tension.set_ylim(0.0, 1.0)
        self.ax_blob_ts.set_xlim(0.0, 1.0)
        self.ax_blob_ts.set_ylim(0.0, 1.0)

        for blob_idx in range(NUM_BLOBS):
            self.blob_trails[blob_idx].set_data([], [])
            self.blob_markers[blob_idx].set_data([], [])
        self.ax_xy.set_xlim(0.0, 1.0)
        self.ax_xy.set_ylim(0.0, 1.0)

        self.time_canvas.draw_idle()
        self.xy_canvas.draw_idle()

    def _set_y_limits(self, axis, series: List[np.ndarray]) -> None:
        finite_values: List[float] = []
        for arr in series:
            values = arr[np.isfinite(arr)]
            if values.size:
                finite_values.extend(values.tolist())

        if not finite_values:
            axis.set_ylim(0.0, 1.0)
            return

        vmin = min(finite_values)
        vmax = max(finite_values)
        if np.isclose(vmin, vmax):
            pad = max(abs(vmin) * 0.05, 1.0)
        else:
            pad = (vmax - vmin) * 0.08
        axis.set_ylim(vmin - pad, vmax + pad)

    def _update_time_series_static(self) -> None:
        data = self.current_data
        if data is None:
            return

        blob_idx = self._selected_blob_index()
        blob_color = BLOB_COLORS[blob_idx]
        x_axis = data.time_values

        self.line_tension_a.set_data(x_axis, data.tension_a)
        self.line_tension_b.set_data(x_axis, data.tension_b)
        self.line_blob_x.set_data(x_axis, data.blob_x[blob_idx])
        self.line_blob_y.set_data(x_axis, data.blob_y[blob_idx])

        self.line_blob_x.set_color(blob_color)
        self.line_blob_y.set_color(blob_color)
        self.line_blob_x.set_label(f"blob{blob_idx}_x")
        self.line_blob_y.set_label(f"blob{blob_idx}_y")
        self.cursor_tension.set_color(blob_color)
        self.cursor_blob.set_color(blob_color)

        self.ax_blob_ts.set_xlabel(data.time_label)

        if data.num_rows > 1:
            xmin = float(np.nanmin(x_axis))
            xmax = float(np.nanmax(x_axis))
            if np.isclose(xmin, xmax):
                xmax = xmin + 1.0
            self.ax_tension.set_xlim(xmin, xmax)
        else:
            center = float(x_axis[0]) if data.num_rows == 1 else 0.0
            self.ax_tension.set_xlim(center - 0.5, center + 0.5)

        self._set_y_limits(self.ax_tension, [data.tension_a, data.tension_b])
        self._set_y_limits(self.ax_blob_ts, [data.blob_x[blob_idx], data.blob_y[blob_idx]])

        self.ax_tension.legend(loc="upper right")
        self.ax_blob_ts.legend(loc="upper right")
        self.time_canvas.draw_idle()

    def _configure_xy_axes(self) -> None:
        data = self.current_data
        if data is None:
            return

        all_x: List[float] = []
        all_y: List[float] = []
        for blob_idx in range(NUM_BLOBS):
            x_vals = data.blob_x[blob_idx]
            y_vals = data.blob_y[blob_idx]
            valid = np.isfinite(x_vals) & np.isfinite(y_vals)
            if np.any(valid):
                all_x.extend(x_vals[valid].tolist())
                all_y.extend(y_vals[valid].tolist())

        if not all_x or not all_y:
            self.ax_xy.set_xlim(0.0, 1.0)
            self.ax_xy.set_ylim(0.0, 1.0)
            self.ax_xy.invert_yaxis()
            self.xy_canvas.draw_idle()
            return

        xmin = min(all_x)
        xmax = max(all_x)
        ymin = min(all_y)
        ymax = max(all_y)

        x_span = xmax - xmin
        y_span = ymax - ymin
        x_pad = max(x_span * 0.05, 1.0)
        y_pad = max(y_span * 0.05, 1.0)

        self.ax_xy.set_xlim(xmin - x_pad, xmax + x_pad)
        self.ax_xy.set_ylim(ymin - y_pad, ymax + y_pad)
        self.ax_xy.set_aspect("equal", adjustable="box")
        self.ax_xy.invert_yaxis()
        self.xy_canvas.draw_idle()

    def _selected_blob_index(self) -> int:
        text = self.blob_var.get().strip()
        if text.startswith("blob"):
            idx_text = text[4:]
            if idx_text.isdigit():
                idx = int(idx_text)
                if 0 <= idx < NUM_BLOBS:
                    return idx
        return 0

    def on_blob_changed(self, event: object = None) -> None:
        if self.current_data is None:
            return
        self._update_time_series_static()
        self._update_frame(self.current_frame, sync_slider=True)

    def on_slider_changed(self, value: str) -> None:
        if self.current_data is None or self.updating_slider:
            return
        try:
            frame = int(float(value))
        except ValueError:
            return
        self._update_frame(frame, sync_slider=False)

    def _update_frame(self, frame_idx: int, sync_slider: bool) -> None:
        data = self.current_data
        if data is None or data.num_rows == 0:
            self.frame_var.set("Frame: 0/0")
            return

        frame_idx = max(0, min(frame_idx, data.num_rows - 1))
        self.current_frame = frame_idx

        if sync_slider:
            self.updating_slider = True
            self.frame_slider.set(frame_idx)
            self.updating_slider = False

        current_x = float(data.time_values[frame_idx])
        self.cursor_tension.set_xdata([current_x, current_x])
        self.cursor_blob.set_xdata([current_x, current_x])

        for blob_idx in range(NUM_BLOBS):
            x_vals = data.blob_x[blob_idx]
            y_vals = data.blob_y[blob_idx]

            trail_x = x_vals[: frame_idx + 1]
            trail_y = y_vals[: frame_idx + 1]
            valid_trail = np.isfinite(trail_x) & np.isfinite(trail_y)
            self.blob_trails[blob_idx].set_data(trail_x[valid_trail], trail_y[valid_trail])

            current_blob_x = x_vals[frame_idx]
            current_blob_y = y_vals[frame_idx]
            if np.isfinite(current_blob_x) and np.isfinite(current_blob_y):
                self.blob_markers[blob_idx].set_data([current_blob_x], [current_blob_y])
            else:
                self.blob_markers[blob_idx].set_data([], [])

        rep_text = "-"
        pulses_text = "-"
        rep_val = data.repetition[frame_idx]
        pulses_val = data.pulses_moved[frame_idx]
        if np.isfinite(rep_val):
            rep_text = str(int(rep_val))
        if np.isfinite(pulses_val):
            pulses_text = str(int(pulses_val))
        phase = data.phase[frame_idx] if frame_idx < len(data.phase) else ""

        self.frame_var.set(f"Frame: {frame_idx + 1}/{data.num_rows}")
        self.status_var.set(
            f"{data.run_id} | rep={rep_text} | phase={phase or '-'} | pulses={pulses_text}"
        )

        self.time_canvas.draw_idle()
        self.xy_canvas.draw_idle()

    def toggle_playback(self) -> None:
        if self.current_data is None or self.current_data.num_rows == 0:
            return

        if self.playing:
            self.stop_playback()
            return

        if self.current_frame >= self.current_data.num_rows - 1:
            self._update_frame(0, sync_slider=True)

        self.playing = True
        self.play_button.configure(text="Pause")
        self._playback_step()

    def _playback_step(self) -> None:
        data = self.current_data
        if not self.playing or data is None:
            return

        if self.current_frame >= data.num_rows - 1:
            self.stop_playback()
            return

        prev_frame = self.current_frame
        next_frame = prev_frame + 1
        self._update_frame(next_frame, sync_slider=True)
        delay_ms = self._compute_delay_ms(prev_frame, next_frame)
        self.playback_after_id = self.root.after(delay_ms, self._playback_step)

    def _compute_delay_ms(self, prev_idx: int, next_idx: int) -> int:
        data = self.current_data
        if data is None:
            return int(DEFAULT_PLAYBACK_DT_S * 1000)

        speed = self.speed_values.get(self.speed_var.get(), 1.0)
        if data.time_mode == "elapsed_s":
            dt = float(data.time_values[next_idx] - data.time_values[prev_idx])
            if (not np.isfinite(dt)) or dt <= 0:
                dt = DEFAULT_PLAYBACK_DT_S
        else:
            dt = DEFAULT_PLAYBACK_DT_S

        dt = min(max(dt, 0.02), 2.0)
        scaled_ms = int((dt / speed) * 1000)
        return max(20, min(scaled_ms, 2000))

    def stop_playback(self) -> None:
        self.playing = False
        self.play_button.configure(text="Play")
        if self.playback_after_id is not None:
            self.root.after_cancel(self.playback_after_id)
            self.playback_after_id = None

    def on_close(self) -> None:
        self.stop_playback()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize experiment CSV runs with blob playback.")
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Directory containing experiment_*_data.csv / experiment2_*_data.csv files (default: %(default)s).",
    )
    parser.add_argument(
        "--initial-experiment",
        default=None,
        help="Run ID to preselect, e.g. experiment_20260416_143000 or experiment2_20260416_143000.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = ExperimentVisualizerApp(
        root=root,
        experiments_dir=Path(args.experiments_dir).expanduser(),
        initial_experiment=args.initial_experiment,
    )
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
