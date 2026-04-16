import csv
import json
import math
import os
import threading
import time
from datetime import datetime
from statistics import median

import cv2
import lgpio
import tkinter as tk
from tkinter import messagebox, ttk

# =================== SCALE CLASS ===================
class HX711Dual:
    PULSES_FOR_NEXT = {
        "A": 1,  # channel A, gain 128
        "B": 2,  # channel B, gain 32
    }

    def __init__(self, dt_pin, sck_pin, gpio_handle=None, ready_timeout=1.0):
        self.dt = dt_pin
        self.sck = sck_pin
        self.ready_timeout = ready_timeout
        self.own_handle = False

        if gpio_handle is None:
            self.h = lgpio.gpiochip_open(0)
            self.own_handle = True
        else:
            self.h = gpio_handle

        lgpio.gpio_claim_input(self.h, self.dt)
        lgpio.gpio_claim_output(self.h, self.sck)
        lgpio.gpio_write(self.h, self.sck, 0)

        self.current_channel = None

    def _wait_ready(self):
        start = time.time()
        while lgpio.gpio_read(self.h, self.dt):
            if time.time() - start > self.ready_timeout:
                raise TimeoutError(f"HX711 DT pin {self.dt} not ready (stays HIGH)")
            time.sleep(0.001)

    def _read_once_set_next(self, next_channel):
        next_channel = next_channel.upper()
        if next_channel not in self.PULSES_FOR_NEXT:
            raise ValueError("next_channel must be 'A' or 'B'")

        self._wait_ready()

        value = 0
        for _ in range(24):
            lgpio.gpio_write(self.h, self.sck, 1)
            value <<= 1
            lgpio.gpio_write(self.h, self.sck, 0)
            if lgpio.gpio_read(self.h, self.dt):
                value += 1

        for _ in range(self.PULSES_FOR_NEXT[next_channel]):
            lgpio.gpio_write(self.h, self.sck, 1)
            lgpio.gpio_write(self.h, self.sck, 0)

        if value & 0x800000:
            value -= 1 << 24

        self.current_channel = next_channel
        return value

    def read_raw(self, channel):
        channel = channel.upper()
        if channel not in self.PULSES_FOR_NEXT:
            raise ValueError("channel must be 'A' or 'B'")

        if self.current_channel != channel:
            # Throw away one sample while switching channel.
            self._read_once_set_next(channel)
        return self._read_once_set_next(channel)

    def tare(self, channel, samples=15):
        return sum(self.read_raw(channel) for _ in range(samples)) / samples

    def get_weight(self, channel, tare_value, calibration):
        raw = self.read_raw(channel)
        return (raw - tare_value) / calibration

    def close(self):
        if self.own_handle:
            lgpio.gpiochip_close(self.h)


# =================== WINCH MOTOR CLASS ===================
class WinchMotor:
    def __init__(self, pul_pin, dir_pin):
        self.pul = pul_pin
        self.dir = dir_pin
        self.h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self.h, self.pul)
        lgpio.gpio_claim_output(self.h, self.dir)
        lgpio.gpio_write(self.h, self.pul, 0)
        lgpio.gpio_write(self.h, self.dir, 0)

    def move(
        self,
        pulses,
        unwind=False,
        step_delay=0.0001,
        ramp_steps=20,
        ramp_factor=2.5,
        should_stop=None,
    ):
        if pulses <= 0:
            return
        lgpio.gpio_write(self.h, self.dir, 1 if unwind else 0)

        # Simple trapezoidal speed profile by pulse delay:
        # start slower -> cruise -> slow down.
        base_delay = max(step_delay, 1e-6)
        start_delay = base_delay * max(ramp_factor, 1.0)
        ramp_n = max(0, min(ramp_steps, pulses // 2))

        for i in range(pulses):
            if should_stop is not None and should_stop():
                break

            if ramp_n > 0 and i < ramp_n:
                ratio = i / ramp_n
                current_delay = start_delay - (start_delay - base_delay) * ratio
            elif ramp_n > 0 and i >= pulses - ramp_n:
                ratio = (pulses - i - 1) / ramp_n
                current_delay = start_delay - (start_delay - base_delay) * ratio
            else:
                current_delay = base_delay

            lgpio.gpio_write(self.h, self.pul, 1)
            time.sleep(current_delay)
            lgpio.gpio_write(self.h, self.pul, 0)
            time.sleep(current_delay)

    def close(self):
        lgpio.gpiochip_close(self.h)


# =================== BLOB TRACKER ===================
NUM_MARKERS = 5
EXPERIMENTS_DIRNAME = "experiments"
CALIBRATION_FILENAME = "scale_calibration.json"
CALIBRATION_FILE_VERSION = 2
MIN_CALIBRATION_RAW_DELTA = 500.0
MIN_MULTIPLIER_ABS = 1e-9
EMA_ALPHA = 0.3
DEFAULT_LINE_BREAK_DROP_G = 15.0
DEFAULT_LINE_BREAK_DROP_PCT = 40.0
DEFAULT_LINE_BREAK_WINDOW = 10
DEFAULT_LINE_BREAK_BREACHES = 2


def get_blob_detector(min_area=30, max_area=600):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.3
    return cv2.SimpleBlobDetector_create(params)


def find_marker_centers(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    keypoints = detector.detect(gray_blur)
    centers = [(int(k.pt[0]), int(k.pt[1])) for k in keypoints]
    return centers


# =================== UI ===================
class WinchUI:
    def __init__(self, root):
        self.root = root
        root.title("Winch + Scales + Blob Logger")
        root.geometry("1000x750")

        # --- SCALES ---
        self.gpio_handle = lgpio.gpiochip_open(0)
        self.scale = HX711Dual(dt_pin=5, sck_pin=6, gpio_handle=self.gpio_handle)
        self.cal_a = 400
        self.cal_b = 400
        self.calibration_path = os.path.join(os.getcwd(), CALIBRATION_FILENAME)
        self.loaded_input_settings = self._default_input_settings()
        self.suspend_settings_save = False
        self.settings_save_after_id = None
        self.startup_calibration_notice = None
        self.startup_calibration_warning_popup = None
        self._load_calibration_settings()
        self.channel_a = "B"
        self.channel_b = "A"
        self.scale_lock = threading.Lock()
        self.tare_a = self._tare_channel(self.channel_a)
        self.tare_b = self._tare_channel(self.channel_b)

        self.scale_reads_blocked_during_motion = False
        self.last_valid_weights = (0.0, 0.0)

        # --- MOTORS ---
        self.motor_a = WinchMotor(pul_pin=17, dir_pin=27)
        self.motor_b = WinchMotor(pul_pin=23, dir_pin=24)

        # --- CAMERA & BLOB DETECTION ---
        self.camera_lock = threading.Lock()
        self.camera_index = 0
        self.camera_backend_name = "unknown"
        self.last_camera_reopen_attempt = 0.0
        self.cap = self._open_camera_capture()
        self.camera_available = self.cap.isOpened()
        self.capture_enabled_runtime = False
        self.preview_enabled_runtime = True

        self.detector = get_blob_detector()
        self.experiment_output_dir = os.path.join(os.getcwd(), EXPERIMENTS_DIRNAME)
        self.experiment_run_id = None
        self.experiment_data_path = None
        self.experiment_setup_path = None
        self.experiment_data_file = None
        self.experiment_setup_file = None
        self.experiment_data_writer = None
        self.experiment_setup_writer = None

        # Blob trackbar values
        self.blob_min_area = 30
        self.blob_max_area = 600

        # Thread-safe UI state (worker writes, main loop renders)
        self.ui_state_lock = threading.Lock()
        self.ui_state = {
            "progress": "Ready",
            "exp_a": 0.0,
            "exp_b": 0.0,
            "progress_pct": 0.0,
            "phase": "Idle",
            "rep_text": "0 / 0",
            "pulse_text": "0 / 0",
            "correction_text": "0",
            "capture_text": "Off",
        }

        self.experiment_running = False
        self.shutdown_requested = False
        self.experiment_thread = None
        self.active_run_mode = None
        self.run_params = {}
        self.ema_state_a = None
        self.ema_state_b = None
        self.line_break_history_a = []
        self.line_break_history_b = []
        self.line_break_consecutive_breaches = 0
        self.line_break_last_reason = ""

        # --- CREATE NOTEBOOK (TABS) ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_calibration = ttk.Frame(self.notebook)
        self.tab_blob = ttk.Frame(self.notebook)
        self.tab_motor = ttk.Frame(self.notebook)
        self.tab_experiment = ttk.Frame(self.notebook)
        self.tab_experiment2 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_calibration, text="Scale Calibration")
        self.notebook.add(self.tab_blob, text="Blob Setup")
        self.notebook.add(self.tab_motor, text="Motor Control")
        self.notebook.add(self.tab_experiment, text="Experiment")
        self.notebook.add(self.tab_experiment2, text="Experiment 2")

        self.build_calibration_tab()
        self.build_blob_tab()
        self.build_motor_tab()
        self.build_experiment_tab()
        self.build_experiment2_tab()
        self._apply_loaded_input_settings()
        self._bind_input_autosave()
        self._save_calibration_settings()

        self.update_weights()
        self.update_blobs()
        self._apply_ui_state()

    # =================== SCALE CALIBRATION TAB ===================
    def build_calibration_tab(self):
        frame = ttk.LabelFrame(self.tab_calibration, text="Scale Readings & Calibration", padding=10)
        frame.pack(padx=10, pady=10, fill="both", expand=True)

        ttk.Label(frame, text="Scale A:").grid(row=0, column=0, sticky="w", pady=5)
        self.label_a = ttk.Label(frame, text="0.00 g", font=("Arial", 14))
        self.label_a.grid(row=0, column=1, sticky="w", padx=5)
        ttk.Button(frame, text="Tare A", command=self.tare_a_func).grid(row=0, column=2, padx=5)

        ttk.Label(frame, text="Scale B:").grid(row=1, column=0, sticky="w", pady=5)
        self.label_b = ttk.Label(frame, text="0.00 g", font=("Arial", 14))
        self.label_b.grid(row=1, column=1, sticky="w", padx=5)
        ttk.Button(frame, text="Tare B", command=self.tare_b_func).grid(row=1, column=2, padx=5)

        calib_frame = ttk.LabelFrame(frame, text="Independent Multiplier Calibration", padding=10)
        calib_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=10)

        ttk.Label(calib_frame, text="Scale A Known Weight (g):").grid(row=0, column=0, sticky="w", pady=4)
        self.entry_known_a = ttk.Entry(calib_frame, width=10)
        self.entry_known_a.grid(row=0, column=1, sticky="w", padx=5)
        self.entry_known_a.insert(0, "100")
        ttk.Button(calib_frame, text="Calibrate A from Known Weight", command=lambda: self._calibrate_scale_from_known_weight("A")).grid(row=0, column=2, padx=5)

        ttk.Label(calib_frame, text="Scale A Multiplier:").grid(row=0, column=3, sticky="w", pady=4, padx=(20, 0))
        self.entry_cal_a = ttk.Entry(calib_frame, width=12)
        self.entry_cal_a.grid(row=0, column=4, sticky="w", padx=5)
        ttk.Button(calib_frame, text="Apply Manual A", command=lambda: self._apply_manual_multiplier("A")).grid(row=0, column=5, padx=5)

        ttk.Label(calib_frame, text="Scale B Known Weight (g):").grid(row=1, column=0, sticky="w", pady=4)
        self.entry_known_b = ttk.Entry(calib_frame, width=10)
        self.entry_known_b.grid(row=1, column=1, sticky="w", padx=5)
        self.entry_known_b.insert(0, "100")
        ttk.Button(calib_frame, text="Calibrate B from Known Weight", command=lambda: self._calibrate_scale_from_known_weight("B")).grid(row=1, column=2, padx=5)

        ttk.Label(calib_frame, text="Scale B Multiplier:").grid(row=1, column=3, sticky="w", pady=4, padx=(20, 0))
        self.entry_cal_b = ttk.Entry(calib_frame, width=12)
        self.entry_cal_b.grid(row=1, column=4, sticky="w", padx=5)
        ttk.Button(calib_frame, text="Apply Manual B", command=lambda: self._apply_manual_multiplier("B")).grid(row=1, column=5, padx=5)

        info_frame = ttk.LabelFrame(frame, text="Calibration Info", padding=10)
        info_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=10)
        info_text = ttk.Label(
            info_frame,
            text=(
                "1) Click Tare on unloaded scale.\n"
                "2) Place known weight and click calibrate.\n"
                "Multiplier = raw_count_delta / known_grams (divisor used in conversion, can be + or -).\n"
                "You can also type multiplier directly and apply."
            ),
            justify="left",
            wraplength=700,
        )
        info_text.pack()

        self.label_cal_status = ttk.Label(frame, text="Calibration status: Ready")
        self.label_cal_status.grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 0))

        self._refresh_calibration_entries()
        if self.startup_calibration_notice is not None:
            msg, is_error = self.startup_calibration_notice
            self._set_calibration_status(msg, is_error=is_error)
            if is_error and self.startup_calibration_warning_popup:
                self.root.after(
                    200,
                    lambda warning=self.startup_calibration_warning_popup: messagebox.showwarning(
                        "Calibration Warning",
                        warning,
                    ),
                )

    def tare_a_func(self):
        if self.experiment_running:
            self._set_calibration_status("Cannot tare while experiment is running.", is_error=True)
            return
        self.tare_a = self._tare_channel(self.channel_a)
        self._set_calibration_status("Scale A tared.")
        messagebox.showinfo("Tare", "Scale A tared")

    def tare_b_func(self):
        if self.experiment_running:
            self._set_calibration_status("Cannot tare while experiment is running.", is_error=True)
            return
        self.tare_b = self._tare_channel(self.channel_b)
        self._set_calibration_status("Scale B tared.")
        messagebox.showinfo("Tare", "Scale B tared")

    def _set_calibration_status(self, message, is_error=False):
        prefix = "Error" if is_error else "OK"
        text = f"Calibration status: [{prefix}] {message}"
        if hasattr(self, "label_cal_status"):
            self.label_cal_status.config(text=text)
        else:
            self.startup_calibration_notice = (text, is_error)

    def _refresh_calibration_entries(self):
        if hasattr(self, "entry_cal_a"):
            self.entry_cal_a.delete(0, tk.END)
            self.entry_cal_a.insert(0, f"{self.cal_a:.6f}")
        if hasattr(self, "entry_cal_b"):
            self.entry_cal_b.delete(0, tk.END)
            self.entry_cal_b.insert(0, f"{self.cal_b:.6f}")

    def _get_scale_channel(self, scale_id):
        if scale_id == "A":
            return self.channel_a
        if scale_id == "B":
            return self.channel_b
        raise ValueError("scale_id must be 'A' or 'B'")

    def _get_scale_tare(self, scale_id):
        if scale_id == "A":
            return self.tare_a
        if scale_id == "B":
            return self.tare_b
        raise ValueError("scale_id must be 'A' or 'B'")

    def _get_scale_multiplier(self, scale_id):
        if scale_id == "A":
            return self.cal_a
        if scale_id == "B":
            return self.cal_b
        raise ValueError("scale_id must be 'A' or 'B'")

    def _set_scale_multiplier(self, scale_id, value):
        if scale_id == "A":
            self.cal_a = value
            return
        if scale_id == "B":
            self.cal_b = value
            return
        raise ValueError("scale_id must be 'A' or 'B'")

    def _default_input_settings(self):
        return {
            "known_weight_a": "100",
            "known_weight_b": "100",
            "manual_cal_a": "400.000000",
            "manual_cal_b": "400.000000",
            "blob_min_area": 30,
            "blob_max_area": 600,
            "motor_pulses": "20",
            "motor_dir_a": "wind",
            "motor_dir_b": "wind",
            "exp_target_a": "50",
            "exp_target_b": "50",
            "exp_move_amp": "20",
            "exp_dwell_ms": "500",
            "exp_reps": "5",
            "exp_capture_pulses": "5",
            "exp_init_adj_step": "1",
            "exp_init_kp": "0.2",
            "exp_adj_step": "1",
            "exp_tolerance": "2",
            "exp_measurement_delay_ms": "200",
            "exp_measurement_samples": "7",
            "exp_sample_interval_ms": "20",
            "exp_stabilization_timeout_s": "10",
            "exp_max_correction_cycles": "80",
            "exp_move_chunk_pulses": "2",
            "exp_line_break_drop_g": f"{DEFAULT_LINE_BREAK_DROP_G:g}",
            "exp_line_break_drop_pct": f"{DEFAULT_LINE_BREAK_DROP_PCT:g}",
            "exp_line_break_window_samples": str(DEFAULT_LINE_BREAK_WINDOW),
            "exp_line_break_consecutive_breaches": str(DEFAULT_LINE_BREAK_BREACHES),
            "exp_enable_capture": True,
            "exp_show_preview": True,
            "exp2_move_amp": "20",
            "exp2_dwell_ms": "500",
            "exp2_reps": "5",
            "exp2_capture_pulses": "5",
            "exp2_move_chunk_pulses": "2",
            "exp2_side_scale_a": "1.0",
            "exp2_side_scale_b": "1.0",
            "exp2_measurement_delay_ms": "200",
            "exp2_measurement_samples": "7",
            "exp2_sample_interval_ms": "20",
            "exp2_line_break_drop_g": f"{DEFAULT_LINE_BREAK_DROP_G:g}",
            "exp2_line_break_drop_pct": f"{DEFAULT_LINE_BREAK_DROP_PCT:g}",
            "exp2_line_break_window_samples": str(DEFAULT_LINE_BREAK_WINDOW),
            "exp2_line_break_consecutive_breaches": str(DEFAULT_LINE_BREAK_BREACHES),
            "exp2_enable_capture": True,
            "exp2_show_preview": True,
        }

    def _to_bool(self, value, default=False):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
        return default

    def _merge_input_settings(self, raw_settings):
        defaults = self._default_input_settings()
        merged = dict(defaults)
        if not isinstance(raw_settings, dict):
            return merged

        for key, default_value in defaults.items():
            if key not in raw_settings:
                continue
            value = raw_settings[key]
            if isinstance(default_value, bool):
                merged[key] = self._to_bool(value, default_value)
            elif isinstance(default_value, int):
                try:
                    merged[key] = int(value)
                except Exception:
                    merged[key] = default_value
            else:
                merged[key] = str(value)
        return merged

    def _set_entry_value(self, entry_widget, value):
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, str(value))

    def _collect_input_settings(self):
        settings = self._default_input_settings()

        if hasattr(self, "entry_known_a"):
            settings["known_weight_a"] = self.entry_known_a.get()
        if hasattr(self, "entry_known_b"):
            settings["known_weight_b"] = self.entry_known_b.get()
        if hasattr(self, "entry_cal_a"):
            settings["manual_cal_a"] = self.entry_cal_a.get()
        if hasattr(self, "entry_cal_b"):
            settings["manual_cal_b"] = self.entry_cal_b.get()

        if hasattr(self, "scale_min_area"):
            settings["blob_min_area"] = int(self.scale_min_area.get())
        if hasattr(self, "scale_max_area"):
            settings["blob_max_area"] = int(self.scale_max_area.get())

        if hasattr(self, "entry_pulses"):
            settings["motor_pulses"] = self.entry_pulses.get()
        if hasattr(self, "var_dir_a"):
            settings["motor_dir_a"] = self.var_dir_a.get()
        if hasattr(self, "var_dir_b"):
            settings["motor_dir_b"] = self.var_dir_b.get()

        if hasattr(self, "entry_target_a"):
            settings["exp_target_a"] = self.entry_target_a.get()
        if hasattr(self, "entry_target_b"):
            settings["exp_target_b"] = self.entry_target_b.get()
        if hasattr(self, "entry_move_amp"):
            settings["exp_move_amp"] = self.entry_move_amp.get()
        if hasattr(self, "entry_dwell"):
            settings["exp_dwell_ms"] = self.entry_dwell.get()
        if hasattr(self, "entry_reps"):
            settings["exp_reps"] = self.entry_reps.get()
        if hasattr(self, "entry_capture_pulses"):
            settings["exp_capture_pulses"] = self.entry_capture_pulses.get()
        if hasattr(self, "entry_init_adj_step"):
            settings["exp_init_adj_step"] = self.entry_init_adj_step.get()
        if hasattr(self, "entry_init_kp"):
            settings["exp_init_kp"] = self.entry_init_kp.get()
        if hasattr(self, "entry_adj_step"):
            settings["exp_adj_step"] = self.entry_adj_step.get()
        if hasattr(self, "entry_tolerance"):
            settings["exp_tolerance"] = self.entry_tolerance.get()
        if hasattr(self, "entry_measurement_delay"):
            settings["exp_measurement_delay_ms"] = self.entry_measurement_delay.get()
        if hasattr(self, "entry_measurement_samples"):
            settings["exp_measurement_samples"] = self.entry_measurement_samples.get()
        if hasattr(self, "entry_sample_interval"):
            settings["exp_sample_interval_ms"] = self.entry_sample_interval.get()
        if hasattr(self, "entry_stabilization_timeout"):
            settings["exp_stabilization_timeout_s"] = self.entry_stabilization_timeout.get()
        if hasattr(self, "entry_max_corrections"):
            settings["exp_max_correction_cycles"] = self.entry_max_corrections.get()
        if hasattr(self, "entry_move_chunk"):
            settings["exp_move_chunk_pulses"] = self.entry_move_chunk.get()
        if hasattr(self, "entry_line_break_drop_g"):
            settings["exp_line_break_drop_g"] = self.entry_line_break_drop_g.get()
        if hasattr(self, "entry_line_break_drop_pct"):
            settings["exp_line_break_drop_pct"] = self.entry_line_break_drop_pct.get()
        if hasattr(self, "entry_line_break_window"):
            settings["exp_line_break_window_samples"] = self.entry_line_break_window.get()
        if hasattr(self, "entry_line_break_breaches"):
            settings["exp_line_break_consecutive_breaches"] = self.entry_line_break_breaches.get()
        if hasattr(self, "var_enable_capture"):
            settings["exp_enable_capture"] = bool(self.var_enable_capture.get())
        if hasattr(self, "var_show_camera_preview"):
            settings["exp_show_preview"] = bool(self.var_show_camera_preview.get())

        if hasattr(self, "entry_move_amp_exp2"):
            settings["exp2_move_amp"] = self.entry_move_amp_exp2.get()
        if hasattr(self, "entry_dwell_exp2"):
            settings["exp2_dwell_ms"] = self.entry_dwell_exp2.get()
        if hasattr(self, "entry_reps_exp2"):
            settings["exp2_reps"] = self.entry_reps_exp2.get()
        if hasattr(self, "entry_capture_pulses_exp2"):
            settings["exp2_capture_pulses"] = self.entry_capture_pulses_exp2.get()
        if hasattr(self, "entry_move_chunk_exp2"):
            settings["exp2_move_chunk_pulses"] = self.entry_move_chunk_exp2.get()
        if hasattr(self, "entry_side_scale_a_exp2"):
            settings["exp2_side_scale_a"] = self.entry_side_scale_a_exp2.get()
        if hasattr(self, "entry_side_scale_b_exp2"):
            settings["exp2_side_scale_b"] = self.entry_side_scale_b_exp2.get()
        if hasattr(self, "entry_measurement_delay_exp2"):
            settings["exp2_measurement_delay_ms"] = self.entry_measurement_delay_exp2.get()
        if hasattr(self, "entry_measurement_samples_exp2"):
            settings["exp2_measurement_samples"] = self.entry_measurement_samples_exp2.get()
        if hasattr(self, "entry_sample_interval_exp2"):
            settings["exp2_sample_interval_ms"] = self.entry_sample_interval_exp2.get()
        if hasattr(self, "entry_line_break_drop_g_exp2"):
            settings["exp2_line_break_drop_g"] = self.entry_line_break_drop_g_exp2.get()
        if hasattr(self, "entry_line_break_drop_pct_exp2"):
            settings["exp2_line_break_drop_pct"] = self.entry_line_break_drop_pct_exp2.get()
        if hasattr(self, "entry_line_break_window_exp2"):
            settings["exp2_line_break_window_samples"] = self.entry_line_break_window_exp2.get()
        if hasattr(self, "entry_line_break_breaches_exp2"):
            settings["exp2_line_break_consecutive_breaches"] = self.entry_line_break_breaches_exp2.get()
        if hasattr(self, "var_enable_capture_exp2"):
            settings["exp2_enable_capture"] = bool(self.var_enable_capture_exp2.get())
        if hasattr(self, "var_show_camera_preview_exp2"):
            settings["exp2_show_preview"] = bool(self.var_show_camera_preview_exp2.get())

        return settings

    def _apply_loaded_input_settings(self):
        settings = self._merge_input_settings(self.loaded_input_settings)
        self.loaded_input_settings = settings

        self.suspend_settings_save = True
        try:
            if hasattr(self, "entry_known_a"):
                self._set_entry_value(self.entry_known_a, settings["known_weight_a"])
            if hasattr(self, "entry_known_b"):
                self._set_entry_value(self.entry_known_b, settings["known_weight_b"])
            if hasattr(self, "entry_cal_a"):
                self._set_entry_value(self.entry_cal_a, settings["manual_cal_a"])
            if hasattr(self, "entry_cal_b"):
                self._set_entry_value(self.entry_cal_b, settings["manual_cal_b"])

            if hasattr(self, "scale_min_area"):
                self.scale_min_area.set(settings["blob_min_area"])
            if hasattr(self, "scale_max_area"):
                self.scale_max_area.set(settings["blob_max_area"])
            if hasattr(self, "scale_min_area") and hasattr(self, "scale_max_area"):
                self.update_blob_params()

            if hasattr(self, "entry_pulses"):
                self._set_entry_value(self.entry_pulses, settings["motor_pulses"])
            if hasattr(self, "var_dir_a"):
                self.var_dir_a.set(settings["motor_dir_a"])
            if hasattr(self, "var_dir_b"):
                self.var_dir_b.set(settings["motor_dir_b"])

            if hasattr(self, "entry_target_a"):
                self._set_entry_value(self.entry_target_a, settings["exp_target_a"])
            if hasattr(self, "entry_target_b"):
                self._set_entry_value(self.entry_target_b, settings["exp_target_b"])
            if hasattr(self, "entry_move_amp"):
                self._set_entry_value(self.entry_move_amp, settings["exp_move_amp"])
            if hasattr(self, "entry_dwell"):
                self._set_entry_value(self.entry_dwell, settings["exp_dwell_ms"])
            if hasattr(self, "entry_reps"):
                self._set_entry_value(self.entry_reps, settings["exp_reps"])
            if hasattr(self, "entry_capture_pulses"):
                self._set_entry_value(self.entry_capture_pulses, settings["exp_capture_pulses"])
            if hasattr(self, "entry_init_adj_step"):
                self._set_entry_value(self.entry_init_adj_step, settings["exp_init_adj_step"])
            if hasattr(self, "entry_init_kp"):
                self._set_entry_value(self.entry_init_kp, settings["exp_init_kp"])
            if hasattr(self, "entry_adj_step"):
                self._set_entry_value(self.entry_adj_step, settings["exp_adj_step"])
            if hasattr(self, "entry_tolerance"):
                self._set_entry_value(self.entry_tolerance, settings["exp_tolerance"])
            if hasattr(self, "entry_measurement_delay"):
                self._set_entry_value(self.entry_measurement_delay, settings["exp_measurement_delay_ms"])
            if hasattr(self, "entry_measurement_samples"):
                self._set_entry_value(self.entry_measurement_samples, settings["exp_measurement_samples"])
            if hasattr(self, "entry_sample_interval"):
                self._set_entry_value(self.entry_sample_interval, settings["exp_sample_interval_ms"])
            if hasattr(self, "entry_stabilization_timeout"):
                self._set_entry_value(self.entry_stabilization_timeout, settings["exp_stabilization_timeout_s"])
            if hasattr(self, "entry_max_corrections"):
                self._set_entry_value(self.entry_max_corrections, settings["exp_max_correction_cycles"])
            if hasattr(self, "entry_move_chunk"):
                self._set_entry_value(self.entry_move_chunk, settings["exp_move_chunk_pulses"])
            if hasattr(self, "entry_line_break_drop_g"):
                self._set_entry_value(self.entry_line_break_drop_g, settings["exp_line_break_drop_g"])
            if hasattr(self, "entry_line_break_drop_pct"):
                self._set_entry_value(self.entry_line_break_drop_pct, settings["exp_line_break_drop_pct"])
            if hasattr(self, "entry_line_break_window"):
                self._set_entry_value(self.entry_line_break_window, settings["exp_line_break_window_samples"])
            if hasattr(self, "entry_line_break_breaches"):
                self._set_entry_value(self.entry_line_break_breaches, settings["exp_line_break_consecutive_breaches"])
            if hasattr(self, "var_enable_capture"):
                self.var_enable_capture.set(self._to_bool(settings["exp_enable_capture"], True))
            if hasattr(self, "var_show_camera_preview"):
                self.var_show_camera_preview.set(self._to_bool(settings["exp_show_preview"], True))

            if hasattr(self, "entry_move_amp_exp2"):
                self._set_entry_value(self.entry_move_amp_exp2, settings["exp2_move_amp"])
            if hasattr(self, "entry_dwell_exp2"):
                self._set_entry_value(self.entry_dwell_exp2, settings["exp2_dwell_ms"])
            if hasattr(self, "entry_reps_exp2"):
                self._set_entry_value(self.entry_reps_exp2, settings["exp2_reps"])
            if hasattr(self, "entry_capture_pulses_exp2"):
                self._set_entry_value(self.entry_capture_pulses_exp2, settings["exp2_capture_pulses"])
            if hasattr(self, "entry_move_chunk_exp2"):
                self._set_entry_value(self.entry_move_chunk_exp2, settings["exp2_move_chunk_pulses"])
            if hasattr(self, "entry_side_scale_a_exp2"):
                self._set_entry_value(self.entry_side_scale_a_exp2, settings["exp2_side_scale_a"])
            if hasattr(self, "entry_side_scale_b_exp2"):
                self._set_entry_value(self.entry_side_scale_b_exp2, settings["exp2_side_scale_b"])
            if hasattr(self, "entry_measurement_delay_exp2"):
                self._set_entry_value(self.entry_measurement_delay_exp2, settings["exp2_measurement_delay_ms"])
            if hasattr(self, "entry_measurement_samples_exp2"):
                self._set_entry_value(self.entry_measurement_samples_exp2, settings["exp2_measurement_samples"])
            if hasattr(self, "entry_sample_interval_exp2"):
                self._set_entry_value(self.entry_sample_interval_exp2, settings["exp2_sample_interval_ms"])
            if hasattr(self, "entry_line_break_drop_g_exp2"):
                self._set_entry_value(self.entry_line_break_drop_g_exp2, settings["exp2_line_break_drop_g"])
            if hasattr(self, "entry_line_break_drop_pct_exp2"):
                self._set_entry_value(self.entry_line_break_drop_pct_exp2, settings["exp2_line_break_drop_pct"])
            if hasattr(self, "entry_line_break_window_exp2"):
                self._set_entry_value(self.entry_line_break_window_exp2, settings["exp2_line_break_window_samples"])
            if hasattr(self, "entry_line_break_breaches_exp2"):
                self._set_entry_value(self.entry_line_break_breaches_exp2, settings["exp2_line_break_consecutive_breaches"])
            if hasattr(self, "var_enable_capture_exp2"):
                self.var_enable_capture_exp2.set(self._to_bool(settings["exp2_enable_capture"], True))
            if hasattr(self, "var_show_camera_preview_exp2"):
                self.var_show_camera_preview_exp2.set(self._to_bool(settings["exp2_show_preview"], True))
        finally:
            self.suspend_settings_save = False

    def _on_input_widget_changed(self, _event=None):
        self._schedule_settings_save()

    def _schedule_settings_save(self, *_):
        if self.suspend_settings_save or self.shutdown_requested:
            return
        if self.settings_save_after_id is not None:
            try:
                self.root.after_cancel(self.settings_save_after_id)
            except Exception:
                pass
        self.settings_save_after_id = self.root.after(250, self._save_settings_from_scheduler)

    def _save_settings_from_scheduler(self):
        self.settings_save_after_id = None
        try:
            self._save_calibration_settings()
        except Exception:
            # Avoid crashing UI on periodic autosave write errors.
            pass

    def _bind_input_autosave(self):
        entry_widgets = [
            self.entry_known_a,
            self.entry_known_b,
            self.entry_cal_a,
            self.entry_cal_b,
            self.entry_pulses,
            self.entry_target_a,
            self.entry_target_b,
            self.entry_move_amp,
            self.entry_dwell,
            self.entry_reps,
            self.entry_capture_pulses,
            self.entry_init_adj_step,
            self.entry_init_kp,
            self.entry_adj_step,
            self.entry_tolerance,
            self.entry_measurement_delay,
            self.entry_measurement_samples,
            self.entry_sample_interval,
            self.entry_stabilization_timeout,
            self.entry_max_corrections,
            self.entry_move_chunk,
            self.entry_line_break_drop_g,
            self.entry_line_break_drop_pct,
            self.entry_line_break_window,
            self.entry_line_break_breaches,
            self.entry_move_amp_exp2,
            self.entry_dwell_exp2,
            self.entry_reps_exp2,
            self.entry_capture_pulses_exp2,
            self.entry_move_chunk_exp2,
            self.entry_side_scale_a_exp2,
            self.entry_side_scale_b_exp2,
            self.entry_measurement_delay_exp2,
            self.entry_measurement_samples_exp2,
            self.entry_sample_interval_exp2,
            self.entry_line_break_drop_g_exp2,
            self.entry_line_break_drop_pct_exp2,
            self.entry_line_break_window_exp2,
            self.entry_line_break_breaches_exp2,
        ]
        for entry_widget in entry_widgets:
            entry_widget.bind("<KeyRelease>", self._on_input_widget_changed)
            entry_widget.bind("<FocusOut>", self._on_input_widget_changed)

        self.var_dir_a.trace_add("write", self._schedule_settings_save)
        self.var_dir_b.trace_add("write", self._schedule_settings_save)
        self.var_enable_capture.trace_add("write", self._schedule_settings_save)
        self.var_show_camera_preview.trace_add("write", self._schedule_settings_save)
        self.var_enable_capture_exp2.trace_add("write", self._schedule_settings_save)
        self.var_show_camera_preview_exp2.trace_add("write", self._schedule_settings_save)

    def _load_calibration_settings(self):
        try:
            with open(self.calibration_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cal_a = float(data["cal_a"])
            cal_b = float(data["cal_b"])
            if not (math.isfinite(cal_a) and math.isfinite(cal_b)):
                raise ValueError("Calibration values must be finite numbers.")
            if abs(cal_a) <= MIN_MULTIPLIER_ABS or abs(cal_b) <= MIN_MULTIPLIER_ABS:
                raise ValueError("Calibration values must be non-zero.")
            self.cal_a = cal_a
            self.cal_b = cal_b
            raw_inputs = data.get("inputs", {})
            self.loaded_input_settings = self._merge_input_settings(raw_inputs)
            if not isinstance(raw_inputs, dict) or "manual_cal_a" not in raw_inputs:
                self.loaded_input_settings["manual_cal_a"] = f"{self.cal_a:.6f}"
            if not isinstance(raw_inputs, dict) or "manual_cal_b" not in raw_inputs:
                self.loaded_input_settings["manual_cal_b"] = f"{self.cal_b:.6f}"
            self.startup_calibration_notice = (f"Loaded calibration from {CALIBRATION_FILENAME}", False)
        except FileNotFoundError:
            self.startup_calibration_notice = (f"Using default calibration values (no {CALIBRATION_FILENAME} found).", False)
            self.startup_calibration_warning_popup = None
            self.loaded_input_settings["manual_cal_a"] = f"{self.cal_a:.6f}"
            self.loaded_input_settings["manual_cal_b"] = f"{self.cal_b:.6f}"
        except Exception as exc:
            self.cal_a = 400.0
            self.cal_b = 400.0
            self.loaded_input_settings = self._default_input_settings()
            self.loaded_input_settings["manual_cal_a"] = f"{self.cal_a:.6f}"
            self.loaded_input_settings["manual_cal_b"] = f"{self.cal_b:.6f}"
            self.startup_calibration_notice = ("Calibration file invalid; fallback to defaults.", True)
            self.startup_calibration_warning_popup = (
                f"Could not load calibration file {CALIBRATION_FILENAME}. "
                f"Default multipliers were used.\n\nDetails: {exc}"
            )

    def _save_calibration_settings(self):
        inputs = self._collect_input_settings()
        self.loaded_input_settings = self._merge_input_settings(inputs)
        payload = {
            "version": CALIBRATION_FILE_VERSION,
            "cal_a": self.cal_a,
            "cal_b": self.cal_b,
            "inputs": self.loaded_input_settings,
        }
        with open(self.calibration_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _collect_stable_raw(self, channel, sample_count=25, sample_interval_ms=20):
        samples = []
        interval_s = max(sample_interval_ms, 0) / 1000.0

        with self.scale_lock:
            for i in range(sample_count):
                samples.append(self.scale.read_raw(channel))
                if i < sample_count - 1 and interval_s > 0:
                    time.sleep(interval_s)

        return float(median(samples))

    def _collect_stable_weight(self, channel, sample_count=25, sample_interval_ms=20):
        if channel == self.channel_a:
            tare_value = self.tare_a
            cal = self.cal_a
        elif channel == self.channel_b:
            tare_value = self.tare_b
            cal = self.cal_b
        else:
            raise ValueError("channel must match one of configured channels")

        samples = []
        interval_s = max(sample_interval_ms, 0) / 1000.0

        with self.scale_lock:
            for i in range(sample_count):
                samples.append(self.scale.get_weight(channel, tare_value, cal))
                if i < sample_count - 1 and interval_s > 0:
                    time.sleep(interval_s)

        return float(median(samples))

    def _calibrate_scale_from_known_weight(self, scale_id):
        if self.experiment_running:
            self._set_calibration_status("Cannot calibrate while experiment is running.", is_error=True)
            return

        try:
            if scale_id == "A":
                known_weight = float(self.entry_known_a.get())
            elif scale_id == "B":
                known_weight = float(self.entry_known_b.get())
            else:
                raise ValueError("Invalid scale ID")
        except Exception:
            self._set_calibration_status(f"Scale {scale_id}: invalid known weight.", is_error=True)
            return

        if not math.isfinite(known_weight) or known_weight <= 0:
            self._set_calibration_status(f"Scale {scale_id}: known weight must be > 0.", is_error=True)
            return

        try:
            channel = self._get_scale_channel(scale_id)
            tare_ref = self._get_scale_tare(scale_id)
            loaded_raw = self._collect_stable_raw(channel, sample_count=25, sample_interval_ms=20)
            raw_delta = loaded_raw - tare_ref
            if abs(raw_delta) <= MIN_CALIBRATION_RAW_DELTA:
                self._set_calibration_status(
                    f"Scale {scale_id}: raw delta too small ({raw_delta:.1f}). Check weight/tare.",
                    is_error=True,
                )
                return

            new_multiplier = raw_delta / known_weight
            if not math.isfinite(new_multiplier) or abs(new_multiplier) <= MIN_MULTIPLIER_ABS:
                self._set_calibration_status(f"Scale {scale_id}: computed multiplier invalid.", is_error=True)
                return

            self._set_scale_multiplier(scale_id, new_multiplier)
            self._refresh_calibration_entries()
            self._save_calibration_settings()

            measured_weight = self._collect_stable_weight(channel, sample_count=7, sample_interval_ms=20)
            self._set_calibration_status(
                f"Scale {scale_id} calibrated. Multiplier={new_multiplier:.6f}, check reading={measured_weight:.2f} g.",
                is_error=False,
            )
        except Exception as e:
            self._set_calibration_status(f"Scale {scale_id} calibration failed: {e}", is_error=True)

    def _apply_manual_multiplier(self, scale_id):
        if self.experiment_running:
            self._set_calibration_status("Cannot apply calibration while experiment is running.", is_error=True)
            return

        try:
            if scale_id == "A":
                value = float(self.entry_cal_a.get())
            elif scale_id == "B":
                value = float(self.entry_cal_b.get())
            else:
                raise ValueError("Invalid scale ID")
        except Exception:
            self._set_calibration_status(f"Scale {scale_id}: invalid multiplier.", is_error=True)
            return

        if not math.isfinite(value) or abs(value) <= MIN_MULTIPLIER_ABS:
            self._set_calibration_status(f"Scale {scale_id}: multiplier must be finite and non-zero.", is_error=True)
            return

        try:
            self._set_scale_multiplier(scale_id, value)
            self._refresh_calibration_entries()
            self._save_calibration_settings()
            self._set_calibration_status(f"Scale {scale_id} manual multiplier applied: {value:.6f}.", is_error=False)
        except Exception as e:
            self._set_calibration_status(f"Scale {scale_id} save failed: {e}", is_error=True)

    def _tare_channel(self, channel, samples=15):
        with self.scale_lock:
            return self.scale.tare(channel, samples=samples)

    def _read_weights_for_control(self):
        with self.scale_lock:
            w_a = self.scale.get_weight(self.channel_a, self.tare_a, self.cal_a)
            w_b = self.scale.get_weight(self.channel_b, self.tare_b, self.cal_b)
        return w_a, w_b

    # =================== BLOB SETUP TAB ===================
    def build_blob_tab(self):
        frame = ttk.LabelFrame(self.tab_blob, text="Blob Detection Setup", padding=10)
        frame.pack(padx=10, pady=10, fill="both", expand=True)

        trackbar_frame = ttk.LabelFrame(frame, text="Detection Parameters", padding=10)
        trackbar_frame.pack(fill="x", pady=10)

        ttk.Label(trackbar_frame, text="Min Area:").grid(row=0, column=0, sticky="w")
        self.scale_min_area = ttk.Scale(trackbar_frame, from_=10, to=500, orient="horizontal")
        self.scale_min_area.grid(row=0, column=1, sticky="ew", padx=5)
        self.label_min_area = ttk.Label(trackbar_frame, text="30")
        self.label_min_area.grid(row=0, column=2, padx=5)

        ttk.Label(trackbar_frame, text="Max Area:").grid(row=1, column=0, sticky="w")
        self.scale_max_area = ttk.Scale(trackbar_frame, from_=50, to=1000, orient="horizontal")
        self.scale_max_area.grid(row=1, column=1, sticky="ew", padx=5)
        self.label_max_area = ttk.Label(trackbar_frame, text="600")
        self.label_max_area.grid(row=1, column=2, padx=5)

        self.scale_min_area.set(30)
        self.scale_max_area.set(600)

        self.scale_min_area.config(command=self.update_blob_params)
        self.scale_max_area.config(command=self.update_blob_params)

        trackbar_frame.columnconfigure(1, weight=1)

        info_frame = ttk.LabelFrame(frame, text="Camera Preview", padding=10)
        info_frame.pack(fill="both", expand=True, pady=10)
        info_text = ttk.Label(
            info_frame,
            text="Live camera feed is displayed in a separate window.\nAdjust sliders above to detect your markers.\nGreen circles show detected blobs.",
            justify="left",
            wraplength=400,
        )
        info_text.pack()

    def update_blob_params(self, value=None):
        self.blob_min_area = int(self.scale_min_area.get())
        self.blob_max_area = int(self.scale_max_area.get())
        self.label_min_area.config(text=str(self.blob_min_area))
        self.label_max_area.config(text=str(self.blob_max_area))
        self.detector = get_blob_detector(min_area=self.blob_min_area, max_area=self.blob_max_area)

        # While tuning blob params, ensure preview is visible and camera is reopened if needed.
        if not self.experiment_running:
            if hasattr(self, "var_show_camera_preview"):
                self.var_show_camera_preview.set(True)
            with self.camera_lock:
                if not self.cap.isOpened():
                    self.cap.release()
                    self.cap = self._open_camera_capture()
                    self.camera_available = self.cap.isOpened()

        self._schedule_settings_save()

    # =================== MOTOR CONTROL TAB ===================
    def build_motor_tab(self):
        frame = ttk.LabelFrame(self.tab_motor, text="Motor Control", padding=10)
        frame.pack(padx=10, pady=10, fill="both", expand=True)

        quick_frame = ttk.LabelFrame(frame, text="Quick Steps (Preset Movements)", padding=10)
        quick_frame.pack(fill="x", pady=10)

        ttk.Label(quick_frame, text="Motor A:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        button_a_frame = ttk.Frame(quick_frame)
        button_a_frame.grid(row=0, column=1, columnspan=3, sticky="ew")
        ttk.Button(button_a_frame, text="20 → (wind)", width=13, command=lambda: self.quick_move_motor_a(20, unwind=False)).pack(side="left", padx=2)
        ttk.Button(button_a_frame, text="20 ← (release)", width=13, command=lambda: self.quick_move_motor_a(20, unwind=True)).pack(side="left", padx=2)
        ttk.Button(button_a_frame, text="100 → (wind)", width=13, command=lambda: self.quick_move_motor_a(100, unwind=False)).pack(side="left", padx=2)
        ttk.Button(button_a_frame, text="100 ← (release)", width=13, command=lambda: self.quick_move_motor_a(100, unwind=True)).pack(side="left", padx=2)

        ttk.Label(quick_frame, text="Motor B:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        button_b_frame = ttk.Frame(quick_frame)
        button_b_frame.grid(row=1, column=1, columnspan=3, sticky="ew")
        ttk.Button(button_b_frame, text="20 → (wind)", width=13, command=lambda: self.quick_move_motor_b(20, unwind=True)).pack(side="left", padx=2)
        ttk.Button(button_b_frame, text="20 ← (release)", width=13, command=lambda: self.quick_move_motor_b(20, unwind=False)).pack(side="left", padx=2)
        ttk.Button(button_b_frame, text="100 → (wind)", width=13, command=lambda: self.quick_move_motor_b(100, unwind=True)).pack(side="left", padx=2)
        ttk.Button(button_b_frame, text="100 ← (release)", width=13, command=lambda: self.quick_move_motor_b(100, unwind=False)).pack(side="left", padx=2)

        custom_frame = ttk.LabelFrame(frame, text="Custom Motion", padding=10)
        custom_frame.pack(fill="x", pady=10)

        ttk.Label(custom_frame, text="Number of Steps:").grid(row=0, column=0, sticky="w", pady=5)
        self.entry_pulses = ttk.Entry(custom_frame, width=15)
        self.entry_pulses.grid(row=0, column=1, sticky="w", padx=5)
        self.entry_pulses.insert(0, "20")

        ttk.Label(custom_frame, text="Motor A Direction:", font=("Arial", 9, "bold")).grid(row=1, column=0, sticky="w", pady=8)
        self.var_dir_a = tk.StringVar(value="wind")
        ttk.Radiobutton(custom_frame, text="Wind (→)", variable=self.var_dir_a, value="wind").grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(custom_frame, text="Release (←)", variable=self.var_dir_a, value="release").grid(row=1, column=2, sticky="w")

        ttk.Label(custom_frame, text="Motor B Direction:", font=("Arial", 9, "bold")).grid(row=2, column=0, sticky="w", pady=8)
        self.var_dir_b = tk.StringVar(value="wind")
        ttk.Radiobutton(custom_frame, text="Wind (→)", variable=self.var_dir_b, value="wind").grid(row=2, column=1, sticky="w")
        ttk.Radiobutton(custom_frame, text="Release (←)", variable=self.var_dir_b, value="release").grid(row=2, column=2, sticky="w")

        button_frame = ttk.Frame(custom_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=15)
        ttk.Button(button_frame, text="Move A", command=self.move_motor_a, width=12).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Move B", command=self.move_motor_b, width=12).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Move Both", command=self.move_both_motors, width=12).pack(side="left", padx=5)

        info_frame = ttk.LabelFrame(frame, text="Legend", padding=10)
        info_frame.pack(fill="x", pady=10)
        info_text = ttk.Label(
            info_frame,
            text="Wind (→): Pulls on string, increases tension\nRelease (←): Releases string, decreases tension",
            justify="left",
            font=("Arial", 9),
        )
        info_text.pack()

    def move_motor_a(self):
        try:
            pulses = int(self.entry_pulses.get())
        except Exception:
            pulses = 20
        unwind = self.var_dir_a.get() == "release"
        self.motor_a.move(pulses, unwind=unwind)

    def move_motor_b(self):
        try:
            pulses = int(self.entry_pulses.get())
        except Exception:
            pulses = 20
        unwind = self.var_dir_b.get() == "wind"
        self.motor_b.move(pulses, unwind=unwind)

    def move_both_motors(self):
        try:
            pulses = int(self.entry_pulses.get())
        except Exception:
            pulses = 20
        unwind_a = self.var_dir_a.get() == "release"
        unwind_b = self.var_dir_b.get() == "wind"
        self._move_motors_interleaved(
            pulses_a=pulses,
            unwind_a=unwind_a,
            pulses_b=pulses,
            unwind_b=unwind_b,
        )

    def quick_move_motor_a(self, pulses, unwind):
        self.motor_a.move(pulses, unwind=unwind)

    def quick_move_motor_b(self, pulses, unwind):
        self.motor_b.move(pulses, unwind=unwind)

    # =================== EXPERIMENT TAB ===================
    def build_experiment_tab(self):
        frame = ttk.Frame(self.tab_experiment, padding=10)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        setup_frame = ttk.LabelFrame(frame, text="Experiment Setup", padding=10)
        setup_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        setup_frame.columnconfigure(1, weight=1)
        setup_frame.columnconfigure(3, weight=1)

        ttk.Label(setup_frame, text="Target Tension A (g):").grid(row=0, column=0, sticky="w", pady=5, padx=10)
        self.entry_target_a = ttk.Entry(setup_frame, width=10)
        self.entry_target_a.grid(row=0, column=1, sticky="w", padx=5)
        self.entry_target_a.insert(0, "50")

        ttk.Label(setup_frame, text="Target Tension B (g):").grid(row=0, column=2, sticky="w", pady=5, padx=10)
        self.entry_target_b = ttk.Entry(setup_frame, width=10)
        self.entry_target_b.grid(row=0, column=3, sticky="w", padx=5)
        self.entry_target_b.insert(0, "50")

        ttk.Label(setup_frame, text="Movement Amplitude (pulses):").grid(row=1, column=0, sticky="w", pady=5, padx=10)
        self.entry_move_amp = ttk.Entry(setup_frame, width=10)
        self.entry_move_amp.grid(row=1, column=1, sticky="w", padx=5)
        self.entry_move_amp.insert(0, "20")

        ttk.Label(setup_frame, text="Dwell Time (ms):").grid(row=1, column=2, sticky="w", pady=5, padx=10)
        self.entry_dwell = ttk.Entry(setup_frame, width=10)
        self.entry_dwell.grid(row=1, column=3, sticky="w", padx=5)
        self.entry_dwell.insert(0, "500")

        ttk.Label(setup_frame, text="Repetitions:").grid(row=2, column=0, sticky="w", pady=5, padx=10)
        self.entry_reps = ttk.Entry(setup_frame, width=10)
        self.entry_reps.grid(row=2, column=1, sticky="w", padx=5)
        self.entry_reps.insert(0, "5")

        ttk.Label(setup_frame, text="Image Capture Every (pulses):").grid(row=2, column=2, sticky="w", pady=5, padx=10)
        self.entry_capture_pulses = ttk.Entry(setup_frame, width=10)
        self.entry_capture_pulses.grid(row=2, column=3, sticky="w", padx=5)
        self.entry_capture_pulses.insert(0, "5")

        # Row 2 of the experiment tab: controls (left) + live status (right).
        control_frame = ttk.LabelFrame(frame, text="Tension Control", padding=10)
        control_frame.grid(row=1, column=0, sticky="nsew", pady=0, padx=(0, 5))

        ttk.Label(control_frame, text="Init Correction Step (pulses):").grid(row=0, column=0, sticky="w")
        self.entry_init_adj_step = ttk.Entry(control_frame, width=10)
        self.entry_init_adj_step.grid(row=0, column=1, sticky="w", padx=5)
        self.entry_init_adj_step.insert(0, "1")

        ttk.Label(control_frame, text="Init Kp (pulses/g):").grid(row=1, column=0, sticky="w")
        self.entry_init_kp = ttk.Entry(control_frame, width=10)
        self.entry_init_kp.grid(row=1, column=1, sticky="w", padx=5)
        self.entry_init_kp.insert(0, "0.2")

        ttk.Label(control_frame, text="Experiment Correction Step (pulses):").grid(row=2, column=0, sticky="w")
        self.entry_adj_step = ttk.Entry(control_frame, width=10)
        self.entry_adj_step.grid(row=2, column=1, sticky="w", padx=5)
        self.entry_adj_step.insert(0, "1")

        ttk.Label(control_frame, text="Tolerance (g):").grid(row=3, column=0, sticky="w")
        self.entry_tolerance = ttk.Entry(control_frame, width=10)
        self.entry_tolerance.grid(row=3, column=1, sticky="w", padx=5)
        self.entry_tolerance.insert(0, "2")

        ttk.Label(control_frame, text="Measurement Delay (ms):").grid(row=4, column=0, sticky="w")
        self.entry_measurement_delay = ttk.Entry(control_frame, width=10)
        self.entry_measurement_delay.grid(row=4, column=1, sticky="w", padx=5)
        self.entry_measurement_delay.insert(0, "200")

        ttk.Label(control_frame, text="Measurement Samples:").grid(row=5, column=0, sticky="w")
        self.entry_measurement_samples = ttk.Entry(control_frame, width=10)
        self.entry_measurement_samples.grid(row=5, column=1, sticky="w", padx=5)
        self.entry_measurement_samples.insert(0, "7")

        ttk.Label(control_frame, text="Sample Interval (ms):").grid(row=6, column=0, sticky="w")
        self.entry_sample_interval = ttk.Entry(control_frame, width=10)
        self.entry_sample_interval.grid(row=6, column=1, sticky="w", padx=5)
        self.entry_sample_interval.insert(0, "20")

        ttk.Label(control_frame, text="Stabilization Timeout (s):").grid(row=7, column=0, sticky="w")
        self.entry_stabilization_timeout = ttk.Entry(control_frame, width=10)
        self.entry_stabilization_timeout.grid(row=7, column=1, sticky="w", padx=5)
        self.entry_stabilization_timeout.insert(0, "10")

        ttk.Label(control_frame, text="Max Experiment Correction Cycles:").grid(row=8, column=0, sticky="w")
        self.entry_max_corrections = ttk.Entry(control_frame, width=10)
        self.entry_max_corrections.grid(row=8, column=1, sticky="w", padx=5)
        self.entry_max_corrections.insert(0, "80")

        ttk.Label(control_frame, text="Move Chunk (pulses):").grid(row=9, column=0, sticky="w")
        self.entry_move_chunk = ttk.Entry(control_frame, width=10)
        self.entry_move_chunk.grid(row=9, column=1, sticky="w", padx=5)
        self.entry_move_chunk.insert(0, "2")

        ttk.Label(control_frame, text="Line-break Drop (g):").grid(row=10, column=0, sticky="w")
        self.entry_line_break_drop_g = ttk.Entry(control_frame, width=10)
        self.entry_line_break_drop_g.grid(row=10, column=1, sticky="w", padx=5)
        self.entry_line_break_drop_g.insert(0, f"{DEFAULT_LINE_BREAK_DROP_G:g}")

        ttk.Label(control_frame, text="Line-break Drop (%):").grid(row=11, column=0, sticky="w")
        self.entry_line_break_drop_pct = ttk.Entry(control_frame, width=10)
        self.entry_line_break_drop_pct.grid(row=11, column=1, sticky="w", padx=5)
        self.entry_line_break_drop_pct.insert(0, f"{DEFAULT_LINE_BREAK_DROP_PCT:g}")

        ttk.Label(control_frame, text="Safety Baseline Window:").grid(row=12, column=0, sticky="w")
        self.entry_line_break_window = ttk.Entry(control_frame, width=10)
        self.entry_line_break_window.grid(row=12, column=1, sticky="w", padx=5)
        self.entry_line_break_window.insert(0, str(DEFAULT_LINE_BREAK_WINDOW))

        ttk.Label(control_frame, text="Safety Consecutive Breaches:").grid(row=13, column=0, sticky="w")
        self.entry_line_break_breaches = ttk.Entry(control_frame, width=10)
        self.entry_line_break_breaches.grid(row=13, column=1, sticky="w", padx=5)
        self.entry_line_break_breaches.insert(0, str(DEFAULT_LINE_BREAK_BREACHES))

        self.var_enable_capture = tk.BooleanVar(value=True)
        self.var_show_camera_preview = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Enable Image Capture", variable=self.var_enable_capture).grid(row=14, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Checkbutton(control_frame, text="Show Camera Preview", variable=self.var_show_camera_preview).grid(row=15, column=0, columnspan=2, sticky="w", pady=3)

        status_frame = ttk.LabelFrame(frame, text="Live Status", padding=10)
        status_frame.grid(row=1, column=1, sticky="nsew", pady=0, padx=(5, 0))

        ttk.Label(status_frame, text="Current A:").grid(row=0, column=0, sticky="w")
        self.label_exp_a = ttk.Label(status_frame, text="0.00 g", font=("Arial", 12, "bold"))
        self.label_exp_a.grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(status_frame, text="Current B:").grid(row=1, column=0, sticky="w")
        self.label_exp_b = ttk.Label(status_frame, text="0.00 g", font=("Arial", 12, "bold"))
        self.label_exp_b.grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(status_frame, text="Progress:").grid(row=2, column=0, sticky="w")
        self.label_progress = ttk.Label(status_frame, text="Ready", font=("Arial", 10))
        self.label_progress.grid(row=2, column=1, sticky="w", padx=5)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(status_frame, orient="horizontal", mode="determinate", maximum=100, variable=self.progress_var)
        self.progress_bar.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 4))

        ttk.Label(status_frame, text="Phase:").grid(row=4, column=0, sticky="w")
        self.label_phase = ttk.Label(status_frame, text="Idle")
        self.label_phase.grid(row=4, column=1, sticky="w", padx=5)

        ttk.Label(status_frame, text="Repetition:").grid(row=5, column=0, sticky="w")
        self.label_rep = ttk.Label(status_frame, text="0 / 0")
        self.label_rep.grid(row=5, column=1, sticky="w", padx=5)

        ttk.Label(status_frame, text="Phase Pulses:").grid(row=6, column=0, sticky="w")
        self.label_phase_pulses = ttk.Label(status_frame, text="0 / 0")
        self.label_phase_pulses.grid(row=6, column=1, sticky="w", padx=5)

        ttk.Label(status_frame, text="Corrections:").grid(row=7, column=0, sticky="w")
        self.label_corrections = ttk.Label(status_frame, text="0")
        self.label_corrections.grid(row=7, column=1, sticky="w", padx=5)

        ttk.Label(status_frame, text="Capture:").grid(row=8, column=0, sticky="w")
        self.label_capture = ttk.Label(status_frame, text="Off")
        self.label_capture.grid(row=8, column=1, sticky="w", padx=5)

        status_frame.columnconfigure(1, weight=1)

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=14, padx=10)
        ttk.Button(button_frame, text="Run Experiment", command=self.run_experiment, width=20).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Stop", command=self.stop_experiment, width=20).pack(side="left", padx=5)

    def build_experiment2_tab(self):
        frame = ttk.Frame(self.tab_experiment2, padding=10)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        setup_frame = ttk.LabelFrame(frame, text="Experiment 2 Setup (Open Loop)", padding=10)
        setup_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        setup_frame.columnconfigure(1, weight=1)
        setup_frame.columnconfigure(3, weight=1)

        ttk.Label(setup_frame, text="Movement Amplitude (pulses):").grid(row=0, column=0, sticky="w", pady=5, padx=10)
        self.entry_move_amp_exp2 = ttk.Entry(setup_frame, width=10)
        self.entry_move_amp_exp2.grid(row=0, column=1, sticky="w", padx=5)
        self.entry_move_amp_exp2.insert(0, "20")

        ttk.Label(setup_frame, text="Motor Step Chunk (pulses):").grid(row=0, column=2, sticky="w", pady=5, padx=10)
        self.entry_move_chunk_exp2 = ttk.Entry(setup_frame, width=10)
        self.entry_move_chunk_exp2.grid(row=0, column=3, sticky="w", padx=5)
        self.entry_move_chunk_exp2.insert(0, "2")

        ttk.Label(setup_frame, text="Dwell Time (ms):").grid(row=1, column=0, sticky="w", pady=5, padx=10)
        self.entry_dwell_exp2 = ttk.Entry(setup_frame, width=10)
        self.entry_dwell_exp2.grid(row=1, column=1, sticky="w", padx=5)
        self.entry_dwell_exp2.insert(0, "500")

        ttk.Label(setup_frame, text="Repetitions:").grid(row=1, column=2, sticky="w", pady=5, padx=10)
        self.entry_reps_exp2 = ttk.Entry(setup_frame, width=10)
        self.entry_reps_exp2.grid(row=1, column=3, sticky="w", padx=5)
        self.entry_reps_exp2.insert(0, "5")

        ttk.Label(setup_frame, text="Image Capture Every (pulses):").grid(row=2, column=0, sticky="w", pady=5, padx=10)
        self.entry_capture_pulses_exp2 = ttk.Entry(setup_frame, width=10)
        self.entry_capture_pulses_exp2.grid(row=2, column=1, sticky="w", padx=5)
        self.entry_capture_pulses_exp2.insert(0, "5")

        ttk.Label(setup_frame, text="Motor A Move Scale:").grid(row=3, column=0, sticky="w", pady=5, padx=10)
        self.entry_side_scale_a_exp2 = ttk.Entry(setup_frame, width=10)
        self.entry_side_scale_a_exp2.grid(row=3, column=1, sticky="w", padx=5)
        self.entry_side_scale_a_exp2.insert(0, "1.0")

        ttk.Label(setup_frame, text="Motor B Move Scale:").grid(row=3, column=2, sticky="w", pady=5, padx=10)
        self.entry_side_scale_b_exp2 = ttk.Entry(setup_frame, width=10)
        self.entry_side_scale_b_exp2.grid(row=3, column=3, sticky="w", padx=5)
        self.entry_side_scale_b_exp2.insert(0, "1.0")

        control_frame = ttk.LabelFrame(frame, text="Sampling and Safety", padding=10)
        control_frame.grid(row=1, column=0, sticky="nsew", pady=0, padx=(0, 5))

        ttk.Label(control_frame, text="Measurement Delay (ms):").grid(row=0, column=0, sticky="w")
        self.entry_measurement_delay_exp2 = ttk.Entry(control_frame, width=10)
        self.entry_measurement_delay_exp2.grid(row=0, column=1, sticky="w", padx=5)
        self.entry_measurement_delay_exp2.insert(0, "200")

        ttk.Label(control_frame, text="Measurement Samples:").grid(row=1, column=0, sticky="w")
        self.entry_measurement_samples_exp2 = ttk.Entry(control_frame, width=10)
        self.entry_measurement_samples_exp2.grid(row=1, column=1, sticky="w", padx=5)
        self.entry_measurement_samples_exp2.insert(0, "7")

        ttk.Label(control_frame, text="Sample Interval (ms):").grid(row=2, column=0, sticky="w")
        self.entry_sample_interval_exp2 = ttk.Entry(control_frame, width=10)
        self.entry_sample_interval_exp2.grid(row=2, column=1, sticky="w", padx=5)
        self.entry_sample_interval_exp2.insert(0, "20")

        ttk.Label(control_frame, text="Line-break Drop (g):").grid(row=3, column=0, sticky="w")
        self.entry_line_break_drop_g_exp2 = ttk.Entry(control_frame, width=10)
        self.entry_line_break_drop_g_exp2.grid(row=3, column=1, sticky="w", padx=5)
        self.entry_line_break_drop_g_exp2.insert(0, f"{DEFAULT_LINE_BREAK_DROP_G:g}")

        ttk.Label(control_frame, text="Line-break Drop (%):").grid(row=4, column=0, sticky="w")
        self.entry_line_break_drop_pct_exp2 = ttk.Entry(control_frame, width=10)
        self.entry_line_break_drop_pct_exp2.grid(row=4, column=1, sticky="w", padx=5)
        self.entry_line_break_drop_pct_exp2.insert(0, f"{DEFAULT_LINE_BREAK_DROP_PCT:g}")

        ttk.Label(control_frame, text="Safety Baseline Window:").grid(row=5, column=0, sticky="w")
        self.entry_line_break_window_exp2 = ttk.Entry(control_frame, width=10)
        self.entry_line_break_window_exp2.grid(row=5, column=1, sticky="w", padx=5)
        self.entry_line_break_window_exp2.insert(0, str(DEFAULT_LINE_BREAK_WINDOW))

        ttk.Label(control_frame, text="Safety Consecutive Breaches:").grid(row=6, column=0, sticky="w")
        self.entry_line_break_breaches_exp2 = ttk.Entry(control_frame, width=10)
        self.entry_line_break_breaches_exp2.grid(row=6, column=1, sticky="w", padx=5)
        self.entry_line_break_breaches_exp2.insert(0, str(DEFAULT_LINE_BREAK_BREACHES))

        self.var_enable_capture_exp2 = tk.BooleanVar(value=True)
        self.var_show_camera_preview_exp2 = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Enable Image Capture", variable=self.var_enable_capture_exp2).grid(row=7, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Checkbutton(control_frame, text="Show Camera Preview", variable=self.var_show_camera_preview_exp2).grid(row=8, column=0, columnspan=2, sticky="w", pady=3)

        status_frame = ttk.LabelFrame(frame, text="Live Status", padding=10)
        status_frame.grid(row=1, column=1, sticky="nsew", pady=0, padx=(5, 0))

        ttk.Label(status_frame, text="Current A:").grid(row=0, column=0, sticky="w")
        self.label_exp2_a = ttk.Label(status_frame, text="0.00 g", font=("Arial", 12, "bold"))
        self.label_exp2_a.grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(status_frame, text="Current B:").grid(row=1, column=0, sticky="w")
        self.label_exp2_b = ttk.Label(status_frame, text="0.00 g", font=("Arial", 12, "bold"))
        self.label_exp2_b.grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(status_frame, text="Progress:").grid(row=2, column=0, sticky="w")
        self.label_progress_exp2 = ttk.Label(status_frame, text="Ready", font=("Arial", 10))
        self.label_progress_exp2.grid(row=2, column=1, sticky="w", padx=5)

        self.progress_var_exp2 = tk.DoubleVar(value=0.0)
        self.progress_bar_exp2 = ttk.Progressbar(status_frame, orient="horizontal", mode="determinate", maximum=100, variable=self.progress_var_exp2)
        self.progress_bar_exp2.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 4))

        ttk.Label(status_frame, text="Phase:").grid(row=4, column=0, sticky="w")
        self.label_phase_exp2 = ttk.Label(status_frame, text="Idle")
        self.label_phase_exp2.grid(row=4, column=1, sticky="w", padx=5)

        ttk.Label(status_frame, text="Repetition:").grid(row=5, column=0, sticky="w")
        self.label_rep_exp2 = ttk.Label(status_frame, text="0 / 0")
        self.label_rep_exp2.grid(row=5, column=1, sticky="w", padx=5)

        ttk.Label(status_frame, text="Phase Pulses:").grid(row=6, column=0, sticky="w")
        self.label_phase_pulses_exp2 = ttk.Label(status_frame, text="0 / 0")
        self.label_phase_pulses_exp2.grid(row=6, column=1, sticky="w", padx=5)

        ttk.Label(status_frame, text="Corrections:").grid(row=7, column=0, sticky="w")
        self.label_corrections_exp2 = ttk.Label(status_frame, text="N/A")
        self.label_corrections_exp2.grid(row=7, column=1, sticky="w", padx=5)

        ttk.Label(status_frame, text="Capture:").grid(row=8, column=0, sticky="w")
        self.label_capture_exp2 = ttk.Label(status_frame, text="Off")
        self.label_capture_exp2.grid(row=8, column=1, sticky="w", padx=5)

        status_frame.columnconfigure(1, weight=1)

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=14, padx=10)
        ttk.Button(button_frame, text="Run Experiment 2", command=self.run_experiment2, width=20).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Stop", command=self.stop_experiment, width=20).pack(side="left", padx=5)

    def _set_ui_state(
        self,
        progress=None,
        exp_a=None,
        exp_b=None,
        progress_pct=None,
        phase=None,
        rep_text=None,
        pulse_text=None,
        correction_text=None,
        capture_text=None,
    ):
        with self.ui_state_lock:
            if progress is not None:
                self.ui_state["progress"] = progress
            if exp_a is not None:
                self.ui_state["exp_a"] = exp_a
            if exp_b is not None:
                self.ui_state["exp_b"] = exp_b
            if progress_pct is not None:
                self.ui_state["progress_pct"] = progress_pct
            if phase is not None:
                self.ui_state["phase"] = phase
            if rep_text is not None:
                self.ui_state["rep_text"] = rep_text
            if pulse_text is not None:
                self.ui_state["pulse_text"] = pulse_text
            if correction_text is not None:
                self.ui_state["correction_text"] = correction_text
            if capture_text is not None:
                self.ui_state["capture_text"] = capture_text

    def _apply_ui_state(self):
        with self.ui_state_lock:
            progress = self.ui_state["progress"]
            exp_a = self.ui_state["exp_a"]
            exp_b = self.ui_state["exp_b"]
            progress_pct = self.ui_state["progress_pct"]
            phase = self.ui_state["phase"]
            rep_text = self.ui_state["rep_text"]
            pulse_text = self.ui_state["pulse_text"]
            correction_text = self.ui_state["correction_text"]
            capture_text = self.ui_state["capture_text"]
        self.label_progress.config(text=str(progress))
        self.label_exp_a.config(text=f"{exp_a:.2f} g")
        self.label_exp_b.config(text=f"{exp_b:.2f} g")
        self.progress_var.set(max(0.0, min(100.0, float(progress_pct))))
        self.label_phase.config(text=str(phase))
        self.label_rep.config(text=str(rep_text))
        self.label_phase_pulses.config(text=str(pulse_text))
        self.label_corrections.config(text=str(correction_text))
        self.label_capture.config(text=str(capture_text))
        if hasattr(self, "label_progress_exp2"):
            self.label_progress_exp2.config(text=str(progress))
            self.label_exp2_a.config(text=f"{exp_a:.2f} g")
            self.label_exp2_b.config(text=f"{exp_b:.2f} g")
            self.progress_var_exp2.set(max(0.0, min(100.0, float(progress_pct))))
            self.label_phase_exp2.config(text=str(phase))
            self.label_rep_exp2.config(text=str(rep_text))
            self.label_phase_pulses_exp2.config(text=str(pulse_text))
            self.label_corrections_exp2.config(text=str(correction_text))
            self.label_capture_exp2.config(text=str(capture_text))
        self.root.after(100, self._apply_ui_state)

    def _set_motion_state(self, is_moving):
        self.scale_reads_blocked_during_motion = is_moving

    def _should_abort_motion(self):
        return self.shutdown_requested or (not self.experiment_running)

    def _calculate_progress_pct(self, rep, phase, phase_pulses=0, phase_total=1):
        reps_total = max(int(self.run_params.get("reps", 1)), 1)
        total_motion_segments = max(reps_total * 2, 1)
        rep = max(0, min(rep, reps_total - 1))
        phase_total = max(int(phase_total), 1)
        phase_pulses = max(0, min(int(phase_pulses), phase_total))

        if phase == "forward":
            segment_index = rep * 2
            fraction = (segment_index + (phase_pulses / phase_total)) / total_motion_segments
        elif phase == "return":
            segment_index = rep * 2 + 1
            fraction = (segment_index + (phase_pulses / phase_total)) / total_motion_segments
        elif phase == "stabilize":
            segment_index = rep * 2
            fraction = segment_index / total_motion_segments
        elif phase == "done":
            fraction = 1.0
        else:
            fraction = 0.0

        return max(0.0, min(100.0, fraction * 100.0))

    def _filter_samples(self, samples_a, samples_b):
        med_a = float(median(samples_a))
        med_b = float(median(samples_b))

        if self.ema_state_a is None:
            self.ema_state_a = med_a
        else:
            self.ema_state_a = self.ema_state_a + EMA_ALPHA * (med_a - self.ema_state_a)

        if self.ema_state_b is None:
            self.ema_state_b = med_b
        else:
            self.ema_state_b = self.ema_state_b + EMA_ALPHA * (med_b - self.ema_state_b)

        return med_a, med_b, self.ema_state_a, self.ema_state_b

    def _collect_settled_samples(self):
        params = self.run_params
        delay_s = params["measurement_delay_ms"] / 1000.0
        interval_s = params["sample_interval_ms"] / 1000.0
        num_samples = params["measurement_samples"]

        if delay_s > 0:
            time.sleep(delay_s)

        samples_a = []
        samples_b = []
        attempts = 0
        max_attempts = max(num_samples * 3, num_samples)

        while len(samples_a) < num_samples and attempts < max_attempts and self.experiment_running:
            attempts += 1
            try:
                w_a, w_b = self._read_weights_for_control()
                samples_a.append(w_a)
                samples_b.append(w_b)
            except Exception:
                pass

            if len(samples_a) < num_samples and interval_s > 0:
                time.sleep(interval_s)

        measurement_valid = len(samples_a) == num_samples
        if not measurement_valid:
            if self.scale_reads_blocked_during_motion:
                self._set_motion_state(False)
            return None

        med_a, med_b, filt_a, filt_b = self._filter_samples(samples_a, samples_b)
        self.last_valid_weights = (filt_a, filt_b)

        if self.scale_reads_blocked_during_motion:
            self._set_motion_state(False)

        return {
            "valid": True,
            "samples_a": samples_a,
            "samples_b": samples_b,
            "median_a": med_a,
            "median_b": med_b,
            "filtered_a": filt_a,
            "filtered_b": filt_b,
        }

    def _reset_line_break_safety_state(self, params):
        self.line_break_history_a = []
        self.line_break_history_b = []
        self.line_break_consecutive_breaches = 0
        self.line_break_last_reason = ""
        # Cache parsed settings for fast checks in control loop.
        self.line_break_drop_g = max(0.0, float(params.get("line_break_drop_g", DEFAULT_LINE_BREAK_DROP_G)))
        self.line_break_drop_pct = max(0.0, float(params.get("line_break_drop_pct", DEFAULT_LINE_BREAK_DROP_PCT)))
        self.line_break_window_samples = max(1, int(params.get("line_break_window_samples", DEFAULT_LINE_BREAK_WINDOW)))
        self.line_break_required_breaches = max(
            1,
            int(params.get("line_break_consecutive_breaches", DEFAULT_LINE_BREAK_BREACHES)),
        )

    def _append_line_break_history(self, w_a, w_b):
        self.line_break_history_a.append(float(w_a))
        self.line_break_history_b.append(float(w_b))
        if len(self.line_break_history_a) > self.line_break_window_samples:
            self.line_break_history_a = self.line_break_history_a[-self.line_break_window_samples :]
        if len(self.line_break_history_b) > self.line_break_window_samples:
            self.line_break_history_b = self.line_break_history_b[-self.line_break_window_samples :]

    def _check_line_break_safety(self, w_a, w_b):
        baseline_a = max(self.line_break_history_a) if self.line_break_history_a else None
        baseline_b = max(self.line_break_history_b) if self.line_break_history_b else None

        breach_a = False
        breach_b = False
        details = []

        if baseline_a is not None and abs(baseline_a) > MIN_MULTIPLIER_ABS:
            drop_a = baseline_a - w_a
            pct_a = (drop_a / abs(baseline_a)) * 100.0
            breach_a = (drop_a >= self.line_break_drop_g) and (pct_a >= self.line_break_drop_pct)
            if breach_a:
                details.append(f"A drop={drop_a:.2f}g ({pct_a:.1f}%) from {baseline_a:.2f}g")

        if baseline_b is not None and abs(baseline_b) > MIN_MULTIPLIER_ABS:
            drop_b = baseline_b - w_b
            pct_b = (drop_b / abs(baseline_b)) * 100.0
            breach_b = (drop_b >= self.line_break_drop_g) and (pct_b >= self.line_break_drop_pct)
            if breach_b:
                details.append(f"B drop={drop_b:.2f}g ({pct_b:.1f}%) from {baseline_b:.2f}g")

        if breach_a or breach_b:
            self.line_break_consecutive_breaches += 1
            self.line_break_last_reason = "; ".join(details)
        else:
            self.line_break_consecutive_breaches = 0

        self._append_line_break_history(w_a, w_b)

        if self.line_break_consecutive_breaches >= self.line_break_required_breaches:
            if not self.line_break_last_reason:
                self.line_break_last_reason = "sudden drop detected"
            return "line_break"
        return None

    def _status_text_for_failure(self, status):
        if status == "line_break":
            if self.line_break_last_reason:
                return f"Stopped: line_break ({self.line_break_last_reason})"
            return "Stopped: line_break"
        return f"Stopped: {status}"

    def _compute_exp2_phase_targets(self, params):
        base_amp = max(int(params["move_amp"]), 0)
        scale_a = float(params.get("side_scale_a", 1.0))
        scale_b = float(params.get("side_scale_b", 1.0))
        target_a = max(1, int(round(base_amp * scale_a))) if base_amp > 0 else 0
        target_b = max(1, int(round(base_amp * scale_b))) if base_amp > 0 else 0
        return target_a, target_b

    def _compute_correction(self, error, tolerance, adj_step):
        if error > tolerance:
            return adj_step
        if error < -tolerance:
            return -adj_step
        return 0

    def _compute_init_correction(self, error, tolerance, min_step, init_kp):
        if abs(error) <= tolerance:
            return 0

        proportional_step = int(round(abs(error) * init_kp))
        pulses = max(int(min_step), proportional_step)
        pulses = max(1, pulses)

        if error > tolerance:
            return pulses
        return -pulses

    def _step_delay_profile(self, pulse_index, total_pulses, base_delay=0.0001, ramp_steps=20, ramp_factor=2.5):
        if total_pulses <= 0:
            return max(base_delay, 1e-6)

        base_delay = max(base_delay, 1e-6)
        start_delay = base_delay * max(ramp_factor, 1.0)
        ramp_n = max(0, min(ramp_steps, total_pulses // 2))

        if ramp_n > 0 and pulse_index < ramp_n:
            ratio = pulse_index / ramp_n
            return start_delay - (start_delay - base_delay) * ratio
        if ramp_n > 0 and pulse_index >= total_pulses - ramp_n:
            ratio = (total_pulses - pulse_index - 1) / ramp_n
            return start_delay - (start_delay - base_delay) * ratio
        return base_delay

    def _move_motors_interleaved(
        self,
        pulses_a,
        unwind_a,
        pulses_b,
        unwind_b,
        step_delay=0.0001,
        ramp_steps=20,
        ramp_factor=2.5,
        should_stop=None,
    ):
        pulses_a = max(0, int(pulses_a))
        pulses_b = max(0, int(pulses_b))
        total = max(pulses_a, pulses_b)
        if total <= 0:
            return

        lgpio.gpio_write(self.motor_a.h, self.motor_a.dir, 1 if unwind_a else 0)
        lgpio.gpio_write(self.motor_b.h, self.motor_b.dir, 1 if unwind_b else 0)

        for i in range(total):
            if should_stop is not None and should_stop():
                break

            current_delay = self._step_delay_profile(i, total, base_delay=step_delay, ramp_steps=ramp_steps, ramp_factor=ramp_factor)
            active_a = i < pulses_a
            active_b = i < pulses_b

            if active_a:
                lgpio.gpio_write(self.motor_a.h, self.motor_a.pul, 1)
            if active_b:
                lgpio.gpio_write(self.motor_b.h, self.motor_b.pul, 1)
            time.sleep(current_delay)

            if active_a:
                lgpio.gpio_write(self.motor_a.h, self.motor_a.pul, 0)
            if active_b:
                lgpio.gpio_write(self.motor_b.h, self.motor_b.pul, 0)
            time.sleep(current_delay)

    def _move_pair_for_phase(self, pulses, phase):
        # Keep experiment mapping centralized.
        if phase == "forward":
            # Forward: A winds, B releases
            self._move_motors_interleaved(
                pulses_a=pulses,
                unwind_a=False,
                pulses_b=pulses,
                unwind_b=False,
                should_stop=self._should_abort_motion,
            )
        else:
            # Return: A releases, B winds
            self._move_motors_interleaved(
                pulses_a=pulses,
                unwind_a=True,
                pulses_b=pulses,
                unwind_b=True,
                should_stop=self._should_abort_motion,
            )

    def _apply_tension_adjustment(self, adj_a, adj_b):
        """
        Positive adjustment means measured tension is too high -> reduce tension.
        Negative adjustment means measured tension is too low -> increase tension.
        Motor A mapping:
          reduce -> unwind=True (release), increase -> unwind=False (wind)
        Motor B mapping is reversed at GPIO level:
          reduce -> unwind=False (release), increase -> unwind=True (wind)
        """
        self._move_motors_interleaved(
            pulses_a=abs(adj_a),
            unwind_a=(adj_a > 0),
            pulses_b=abs(adj_b),
            unwind_b=(adj_b < 0),
            should_stop=self._should_abort_motion,
        )

    def _start_experiment_logging(self, params, run_prefix="experiment"):
        self._close_experiment_logging()
        os.makedirs(self.experiment_output_dir, exist_ok=True)

        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_run_id = f"{run_prefix}_{run_stamp}"
        data_name = f"{self.experiment_run_id}_data.csv"
        setup_name = f"{self.experiment_run_id}_setup.csv"
        self.experiment_data_path = os.path.join(self.experiment_output_dir, data_name)
        self.experiment_setup_path = os.path.join(self.experiment_output_dir, setup_name)

        try:
            self.experiment_data_file = open(self.experiment_data_path, "w", newline="", encoding="utf-8")
            self.experiment_data_writer = csv.writer(self.experiment_data_file)
            data_header = [
                "run_id",
                "timestamp_local",
                "repetition",
                "phase",
                "pulses_moved",
                "tension_a_g",
                "tension_b_g",
            ] + [f"blob{i}_{coord}" for i in range(NUM_MARKERS) for coord in ["x", "y"]]
            self.experiment_data_writer.writerow(data_header)
            self.experiment_data_file.flush()

            self.experiment_setup_file = open(self.experiment_setup_path, "w", newline="", encoding="utf-8")
            self.experiment_setup_writer = csv.writer(self.experiment_setup_file)
            self.experiment_setup_writer.writerow(["key", "value"])

            now_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.experiment_setup_writer.writerow(["run_id", self.experiment_run_id])
            self.experiment_setup_writer.writerow(["run_started_at_local", now_local])
            self.experiment_setup_writer.writerow(["output_dir", self.experiment_output_dir])
            self.experiment_setup_writer.writerow(["data_filename", os.path.basename(self.experiment_data_path)])
            self.experiment_setup_writer.writerow(["setup_filename", os.path.basename(self.experiment_setup_path)])

            for key in sorted(params.keys()):
                self.experiment_setup_writer.writerow([f"param_{key}", params[key]])

            with self.camera_lock:
                camera_open = self.cap.isOpened()
            self.experiment_setup_writer.writerow(["runtime_capture_enabled", int(bool(self.capture_enabled_runtime))])
            self.experiment_setup_writer.writerow(["runtime_preview_enabled", int(bool(self.preview_enabled_runtime))])
            self.experiment_setup_writer.writerow(["runtime_camera_open", int(bool(camera_open))])
            self.experiment_setup_writer.writerow(["runtime_camera_backend", self.camera_backend_name])
            self.experiment_setup_writer.writerow(["runtime_camera_available_flag", int(bool(self.camera_available))])

            self.experiment_setup_writer.writerow(["cal_a", self.cal_a])
            self.experiment_setup_writer.writerow(["cal_b", self.cal_b])
            self.experiment_setup_writer.writerow(["tare_a", self.tare_a])
            self.experiment_setup_writer.writerow(["tare_b", self.tare_b])
            self.experiment_setup_writer.writerow(["channel_a", self.channel_a])
            self.experiment_setup_writer.writerow(["channel_b", self.channel_b])
            self.experiment_setup_file.flush()
        except Exception:
            self._close_experiment_logging()
            raise

    def _log_experiment_data_row(self, rep, phase, pulses_moved, w_a, w_b, centers=None):
        if self.experiment_data_writer is None:
            return

        timestamp_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        safe_centers = centers or []
        row = [
            self.experiment_run_id or "",
            timestamp_local,
            rep + 1,
            phase,
            pulses_moved,
            round(w_a, 3),
            round(w_b, 3),
        ]

        for i in range(NUM_MARKERS):
            if i < len(safe_centers):
                row += [safe_centers[i][0], safe_centers[i][1]]
            else:
                row += ["", ""]

        self.experiment_data_writer.writerow(row)
        self.experiment_data_file.flush()

    def _log_capture_data_row(self, rep, phase, pulses_moved, w_a, w_b, centers):
        self._log_experiment_data_row(rep, phase, pulses_moved, w_a, w_b, centers=centers)

    def _close_experiment_logging(self):
        if self.experiment_data_file is not None:
            try:
                self.experiment_data_file.close()
            except Exception:
                pass
        if self.experiment_setup_file is not None:
            try:
                self.experiment_setup_file.close()
            except Exception:
                pass

        self.experiment_data_file = None
        self.experiment_setup_file = None
        self.experiment_data_writer = None
        self.experiment_setup_writer = None
        self.experiment_data_path = None
        self.experiment_setup_path = None
        self.experiment_run_id = None

    def _capture_and_log_blobs(self):
        if not self.capture_enabled_runtime:
            return False, []
        with self.camera_lock:
            if not self.cap.isOpened():
                return False, []
            ret, frame = self.cap.read()
            if not ret:
                return False, []

        centers = find_marker_centers(frame, self.detector)

        return True, centers

    def _get_experiment_params(self):
        return {
            "target_a": float(self.entry_target_a.get()),
            "target_b": float(self.entry_target_b.get()),
            "move_amp": int(self.entry_move_amp.get()),
            "dwell_ms": int(self.entry_dwell.get()),
            "reps": int(self.entry_reps.get()),
            "init_adj_step": int(self.entry_init_adj_step.get()),
            "init_kp": float(self.entry_init_kp.get()),
            "adj_step": int(self.entry_adj_step.get()),
            "tolerance": float(self.entry_tolerance.get()),
            "capture_pulses": int(self.entry_capture_pulses.get()),
            "measurement_delay_ms": int(self.entry_measurement_delay.get()),
            "measurement_samples": int(self.entry_measurement_samples.get()),
            "sample_interval_ms": int(self.entry_sample_interval.get()),
            "stabilization_timeout_s": float(self.entry_stabilization_timeout.get()),
            "max_correction_cycles": int(self.entry_max_corrections.get()),
            "move_chunk_pulses": int(self.entry_move_chunk.get()),
            "line_break_drop_g": float(self.entry_line_break_drop_g.get()),
            "line_break_drop_pct": float(self.entry_line_break_drop_pct.get()),
            "line_break_window_samples": int(self.entry_line_break_window.get()),
            "line_break_consecutive_breaches": int(self.entry_line_break_breaches.get()),
            "enable_capture": bool(self.var_enable_capture.get()),
            "show_preview": bool(self.var_show_camera_preview.get()),
        }

    def _get_experiment2_params(self):
        return {
            "move_amp": int(self.entry_move_amp_exp2.get()),
            "dwell_ms": int(self.entry_dwell_exp2.get()),
            "reps": int(self.entry_reps_exp2.get()),
            "capture_pulses": int(self.entry_capture_pulses_exp2.get()),
            "move_chunk_pulses": int(self.entry_move_chunk_exp2.get()),
            "side_scale_a": float(self.entry_side_scale_a_exp2.get()),
            "side_scale_b": float(self.entry_side_scale_b_exp2.get()),
            "measurement_delay_ms": int(self.entry_measurement_delay_exp2.get()),
            "measurement_samples": int(self.entry_measurement_samples_exp2.get()),
            "sample_interval_ms": int(self.entry_sample_interval_exp2.get()),
            "line_break_drop_g": float(self.entry_line_break_drop_g_exp2.get()),
            "line_break_drop_pct": float(self.entry_line_break_drop_pct_exp2.get()),
            "line_break_window_samples": int(self.entry_line_break_window_exp2.get()),
            "line_break_consecutive_breaches": int(self.entry_line_break_breaches_exp2.get()),
            "enable_capture": bool(self.var_enable_capture_exp2.get()),
            "show_preview": bool(self.var_show_camera_preview_exp2.get()),
        }

    def _validate_experiment_params(self, params):
        if params["move_amp"] <= 0 or params["reps"] <= 0:
            return "Amplitude and repetitions must be > 0"
        if params["init_adj_step"] <= 0:
            return "Init correction step must be > 0"
        if params["init_kp"] < 0:
            return "Init Kp must be >= 0"
        if params["adj_step"] <= 0:
            return "Experiment correction step must be > 0"
        if params["tolerance"] < 0:
            return "Tolerance must be >= 0"
        if params["dwell_ms"] < 0:
            return "Dwell time must be >= 0"
        if params["measurement_delay_ms"] < 0:
            return "Measurement delay must be >= 0"
        if params["measurement_samples"] <= 0:
            return "Measurement samples must be > 0"
        if params["sample_interval_ms"] < 0:
            return "Sample interval must be >= 0"
        if params["stabilization_timeout_s"] <= 0:
            return "Stabilization timeout must be > 0"
        if params["max_correction_cycles"] <= 0:
            return "Max correction cycles must be > 0"
        if params["move_chunk_pulses"] <= 0:
            return "Move chunk must be > 0"
        if params["line_break_drop_g"] <= 0:
            return "Line-break Drop (g) must be > 0"
        if params["line_break_drop_pct"] <= 0:
            return "Line-break Drop (%) must be > 0"
        if params["line_break_window_samples"] <= 0:
            return "Safety Baseline Window must be > 0"
        if params["line_break_consecutive_breaches"] <= 0:
            return "Safety Consecutive Breaches must be > 0"
        if params["enable_capture"] and params["capture_pulses"] <= 0:
            return "Image Capture Every (pulses) must be > 0 when capture is enabled"
        return None

    def _validate_experiment2_params(self, params):
        if params["move_amp"] <= 0 or params["reps"] <= 0:
            return "Amplitude and repetitions must be > 0"
        if params["dwell_ms"] < 0:
            return "Dwell time must be >= 0"
        if params["move_chunk_pulses"] <= 0:
            return "Motor step chunk must be > 0"
        if (not math.isfinite(params["side_scale_a"])) or params["side_scale_a"] <= 0:
            return "Motor A Move Scale must be a finite value > 0"
        if (not math.isfinite(params["side_scale_b"])) or params["side_scale_b"] <= 0:
            return "Motor B Move Scale must be a finite value > 0"
        if params["measurement_delay_ms"] < 0:
            return "Measurement delay must be >= 0"
        if params["measurement_samples"] <= 0:
            return "Measurement samples must be > 0"
        if params["sample_interval_ms"] < 0:
            return "Sample interval must be >= 0"
        if params["line_break_drop_g"] <= 0:
            return "Line-break Drop (g) must be > 0"
        if params["line_break_drop_pct"] <= 0:
            return "Line-break Drop (%) must be > 0"
        if params["line_break_window_samples"] <= 0:
            return "Safety Baseline Window must be > 0"
        if params["line_break_consecutive_breaches"] <= 0:
            return "Safety Consecutive Breaches must be > 0"
        if params["enable_capture"] and params["capture_pulses"] <= 0:
            return "Image Capture Every (pulses) must be > 0 when capture is enabled"
        return None

    def _open_camera_capture(self):
        backend_candidates = []
        if hasattr(cv2, "CAP_V4L2"):
            backend_candidates.append(("V4L2", cv2.CAP_V4L2))
        if hasattr(cv2, "CAP_GSTREAMER"):
            backend_candidates.append(("GSTREAMER", cv2.CAP_GSTREAMER))
        if hasattr(cv2, "CAP_ANY"):
            backend_candidates.append(("ANY", cv2.CAP_ANY))
        else:
            backend_candidates.append(("DEFAULT", None))

        tried = set()
        for backend_name, backend_id in backend_candidates:
            if backend_name in tried:
                continue
            tried.add(backend_name)

            if backend_id is None:
                cap = cv2.VideoCapture(self.camera_index)
            else:
                cap = cv2.VideoCapture(self.camera_index, backend_id)

            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                self.camera_backend_name = backend_name
                return cap
            cap.release()

        self.camera_backend_name = "unavailable"
        return cv2.VideoCapture()

    def _resolve_capture_mode_for_run(self, params):
        capture_requested = params["enable_capture"]
        preview_requested = params["show_preview"]

        self.capture_enabled_runtime = False
        self.preview_enabled_runtime = preview_requested

        if not capture_requested:
            return True

        # Refresh camera availability at run start.
        with self.camera_lock:
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = self._open_camera_capture()
            self.camera_available = self.cap.isOpened()

        if self.camera_available:
            self.capture_enabled_runtime = True
            return True

        proceed_without_capture = messagebox.askyesno(
            "Camera unavailable",
            "Image capture is enabled but camera is not available. Continue experiment without image capture?",
        )
        if not proceed_without_capture:
            return False

        self.capture_enabled_runtime = False
        return True

    def stop_experiment(self):
        self.experiment_running = False
        self.active_run_mode = None
        self._set_ui_state(progress="Stopped", phase="Stopped")

    def run_experiment(self):
        if self.experiment_running:
            return

        try:
            params = self._get_experiment_params()
        except Exception:
            messagebox.showerror("Error", "Invalid input parameters")
            return

        err = self._validate_experiment_params(params)
        if err:
            messagebox.showerror("Error", err)
            return

        if not self._resolve_capture_mode_for_run(params):
            return

        try:
            self._start_experiment_logging(params, run_prefix="experiment")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize experiment log files: {e}")
            return

        self.run_params = params
        self.active_run_mode = "experiment"
        self.ema_state_a = None
        self.ema_state_b = None
        self._reset_line_break_safety_state(params)

        self.experiment_running = True
        self.shutdown_requested = False
        capture_state = "On" if self.capture_enabled_runtime else "Off"
        self._set_ui_state(
            progress="Initializing...",
            progress_pct=0.0,
            phase="Init",
            rep_text=f"0 / {params['reps']}",
            pulse_text=f"0 / {params['move_amp']}",
            correction_text="0",
            capture_text=capture_state,
        )

        self.experiment_thread = threading.Thread(target=self._run_experiment_thread, daemon=True)
        self.experiment_thread.start()

    def run_experiment2(self):
        if self.experiment_running:
            return

        try:
            params = self._get_experiment2_params()
        except Exception:
            messagebox.showerror("Error", "Invalid Experiment 2 input parameters")
            return

        err = self._validate_experiment2_params(params)
        if err:
            messagebox.showerror("Error", err)
            return

        if not self._resolve_capture_mode_for_run(params):
            return

        try:
            self._start_experiment_logging(params, run_prefix="experiment2")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize Experiment 2 log files: {e}")
            return

        self.run_params = params
        self.active_run_mode = "experiment2"
        self.ema_state_a = None
        self.ema_state_b = None
        self._reset_line_break_safety_state(params)

        self.experiment_running = True
        self.shutdown_requested = False
        capture_state = "On" if self.capture_enabled_runtime else "Off"
        target_a, target_b = self._compute_exp2_phase_targets(params)
        phase_total = max(target_a, target_b)
        self._set_ui_state(
            progress="Initializing Experiment 2...",
            progress_pct=0.0,
            phase="Init",
            rep_text=f"0 / {params['reps']}",
            pulse_text=f"0 / {phase_total}",
            correction_text="N/A",
            capture_text=capture_state,
        )

        self.experiment_thread = threading.Thread(target=self._run_experiment2_thread, daemon=True)
        self.experiment_thread.start()

    def _stabilize_tension(self, rep):
        params = self.run_params
        start_time = time.time()
        init_correction_cycles = 0

        while self.experiment_running:
            if time.time() - start_time >= params["stabilization_timeout_s"]:
                return "timeout"

            measurement = self._collect_settled_samples()
            if not measurement:
                continue

            w_a = measurement["filtered_a"]
            w_b = measurement["filtered_b"]
            safety_status = self._check_line_break_safety(w_a, w_b)
            if safety_status is not None:
                return safety_status
            self._set_ui_state(
                progress=f"Repetition {rep+1}/{params['reps']} - Stabilizing",
                exp_a=w_a,
                exp_b=w_b,
                progress_pct=self._calculate_progress_pct(rep, "stabilize"),
                phase="Stabilizing",
                rep_text=f"{rep+1} / {params['reps']}",
                pulse_text=f"0 / {params['move_amp']}",
                # Init corrections are prep-only and do not count in experiment correction total.
                correction_text="0",
            )

            err_a = w_a - params["target_a"]
            err_b = w_b - params["target_b"]

            adj_a = self._compute_init_correction(err_a, params["tolerance"], params["init_adj_step"], params["init_kp"])
            adj_b = self._compute_init_correction(err_b, params["tolerance"], params["init_adj_step"], params["init_kp"])

            if adj_a == 0 and adj_b == 0:
                return "ok"

            self._set_motion_state(True)
            self._apply_tension_adjustment(adj_a, adj_b)
            init_correction_cycles += 1

        return "stopped"

    def _move_with_tension_feedback(self, rep, phase):
        params = self.run_params
        pulses_left = params["move_amp"]
        chunk = max(params["move_chunk_pulses"], 1)
        pulses_moved = 0
        correction_cycles = 0

        while pulses_left > 0 and self.experiment_running:
            if correction_cycles >= params["max_correction_cycles"]:
                return "max_corrections"

            step_pulses = min(chunk, pulses_left)
            self._set_motion_state(True)
            self._move_pair_for_phase(step_pulses, phase)
            pulses_moved += step_pulses
            pulses_left -= step_pulses

            measurement = self._collect_settled_samples()
            if not measurement:
                continue

            w_a = measurement["filtered_a"]
            w_b = measurement["filtered_b"]
            safety_status = self._check_line_break_safety(w_a, w_b)
            if safety_status is not None:
                return safety_status
            self._set_ui_state(
                progress=f"Repetition {rep+1}/{params['reps']} - {phase.title()}",
                exp_a=w_a,
                exp_b=w_b,
                progress_pct=self._calculate_progress_pct(rep, phase, pulses_moved, params["move_amp"]),
                phase=phase.title(),
                rep_text=f"{rep+1} / {params['reps']}",
                pulse_text=f"{pulses_moved} / {params['move_amp']}",
                correction_text=str(correction_cycles),
            )

            err_a = w_a - params["target_a"]
            err_b = w_b - params["target_b"]
            adj_a = self._compute_correction(err_a, params["tolerance"], params["adj_step"])
            adj_b = self._compute_correction(err_b, params["tolerance"], params["adj_step"])

            if adj_a != 0 or adj_b != 0:
                self._set_motion_state(True)
                self._apply_tension_adjustment(adj_a, adj_b)
                correction_cycles += 1
                self._set_ui_state(correction_text=str(correction_cycles))

                correction_measurement = self._collect_settled_samples()
                if correction_measurement:
                    w_a = correction_measurement["filtered_a"]
                    w_b = correction_measurement["filtered_b"]
                    safety_status = self._check_line_break_safety(w_a, w_b)
                    if safety_status is not None:
                        return safety_status
                    self._set_ui_state(exp_a=w_a, exp_b=w_b)

            capture_ok = False
            centers = []
            if self.capture_enabled_runtime and pulses_moved % params["capture_pulses"] == 0:
                capture_ok, centers = self._capture_and_log_blobs()
                if capture_ok:
                    self._log_capture_data_row(rep, phase, pulses_moved, w_a, w_b, centers)

        return "ok" if self.experiment_running else "stopped"

    def _run_experiment_thread(self):
        try:
            params = self.run_params
            dwell_sec = params["dwell_ms"] / 1000.0

            for rep in range(params["reps"]):
                if not self.experiment_running:
                    break

                self._set_ui_state(progress=f"Repetition {rep+1}/{params['reps']} - Stabilizing")
                status = self._stabilize_tension(rep)
                if status != "ok":
                    self.experiment_running = False
                    self._set_ui_state(progress=self._status_text_for_failure(status), phase="Failed")
                    return

                self._set_ui_state(progress=f"Repetition {rep+1}/{params['reps']} - Forward")
                status = self._move_with_tension_feedback(rep, "forward")
                if status != "ok":
                    self.experiment_running = False
                    self._set_ui_state(progress=self._status_text_for_failure(status), phase="Failed")
                    return
                time.sleep(dwell_sec)

                if not self.experiment_running:
                    break

                self._set_ui_state(progress=f"Repetition {rep+1}/{params['reps']} - Return")
                status = self._move_with_tension_feedback(rep, "return")
                if status != "ok":
                    self.experiment_running = False
                    self._set_ui_state(progress=self._status_text_for_failure(status), phase="Failed")
                    return
                time.sleep(dwell_sec)

            self.experiment_running = False
            self._set_ui_state(
                progress="Experiment Complete!",
                progress_pct=100.0,
                phase="Done",
                rep_text=f"{params['reps']} / {params['reps']}",
            )
        finally:
            self._set_motion_state(False)
            self._close_experiment_logging()
            self.experiment_thread = None
            self.active_run_mode = None

    def _move_open_loop_phase(self, rep, phase):
        params = self.run_params
        target_a, target_b = self._compute_exp2_phase_targets(params)
        pulses_left_a = target_a
        pulses_left_b = target_b
        chunk = max(params["move_chunk_pulses"], 1)
        pulses_moved_a = 0
        pulses_moved_b = 0
        phase_total = max(target_a, target_b)

        while (pulses_left_a > 0 or pulses_left_b > 0) and self.experiment_running:
            step_pulses_a = min(chunk, pulses_left_a) if pulses_left_a > 0 else 0
            step_pulses_b = min(chunk, pulses_left_b) if pulses_left_b > 0 else 0
            self._set_motion_state(True)
            if phase == "forward":
                self._move_motors_interleaved(
                    pulses_a=step_pulses_a,
                    unwind_a=False,
                    pulses_b=step_pulses_b,
                    unwind_b=False,
                    should_stop=self._should_abort_motion,
                )
            else:
                self._move_motors_interleaved(
                    pulses_a=step_pulses_a,
                    unwind_a=True,
                    pulses_b=step_pulses_b,
                    unwind_b=True,
                    should_stop=self._should_abort_motion,
                )
            pulses_moved_a += step_pulses_a
            pulses_moved_b += step_pulses_b
            pulses_left_a -= step_pulses_a
            pulses_left_b -= step_pulses_b
            phase_pulses_moved = max(pulses_moved_a, pulses_moved_b)

            measurement = self._collect_settled_samples()
            if not measurement:
                continue

            w_a = measurement["filtered_a"]
            w_b = measurement["filtered_b"]

            safety_status = self._check_line_break_safety(w_a, w_b)
            if safety_status is not None:
                return safety_status

            capture_ok = False
            centers = []
            if (
                self.capture_enabled_runtime
                and phase_pulses_moved > 0
                and phase_pulses_moved % params["capture_pulses"] == 0
            ):
                capture_ok, centers = self._capture_and_log_blobs()

            self._log_experiment_data_row(
                rep=rep,
                phase=phase,
                pulses_moved=phase_pulses_moved,
                w_a=w_a,
                w_b=w_b,
                centers=centers if capture_ok else [],
            )

            self._set_ui_state(
                progress=f"Experiment 2 - Repetition {rep+1}/{params['reps']} - {phase.title()}",
                exp_a=w_a,
                exp_b=w_b,
                progress_pct=self._calculate_progress_pct(rep, phase, phase_pulses_moved, phase_total),
                phase=phase.title(),
                rep_text=f"{rep+1} / {params['reps']}",
                pulse_text=f"{phase_pulses_moved} / {phase_total}",
                correction_text="N/A",
            )

        return "ok" if self.experiment_running else "stopped"

    def _run_experiment2_thread(self):
        try:
            params = self.run_params
            dwell_sec = params["dwell_ms"] / 1000.0

            for rep in range(params["reps"]):
                if not self.experiment_running:
                    break

                self._set_ui_state(progress=f"Experiment 2 - Repetition {rep+1}/{params['reps']} - Forward")
                status = self._move_open_loop_phase(rep, "forward")
                if status != "ok":
                    self.experiment_running = False
                    self._set_ui_state(progress=self._status_text_for_failure(status), phase="Failed")
                    return
                time.sleep(dwell_sec)

                if not self.experiment_running:
                    break

                self._set_ui_state(progress=f"Experiment 2 - Repetition {rep+1}/{params['reps']} - Return")
                status = self._move_open_loop_phase(rep, "return")
                if status != "ok":
                    self.experiment_running = False
                    self._set_ui_state(progress=self._status_text_for_failure(status), phase="Failed")
                    return
                time.sleep(dwell_sec)

            self.experiment_running = False
            self._set_ui_state(
                progress="Experiment 2 Complete!",
                progress_pct=100.0,
                phase="Done",
                rep_text=f"{params['reps']} / {params['reps']}",
                correction_text="N/A",
            )
        finally:
            self._set_motion_state(False)
            self._close_experiment_logging()
            self.experiment_thread = None
            self.active_run_mode = None

    def update_weights(self):
        try:
            # During experiment, show the last validated control measurement to avoid
            # interfering with settled-sample timing on the HX711 bus.
            if self.experiment_running or self.scale_reads_blocked_during_motion:
                w_a, w_b = self.last_valid_weights
            else:
                w_a, w_b = self._read_weights_for_control()
                self.last_valid_weights = (w_a, w_b)
        except Exception:
            w_a = w_b = 0.0

        self.label_a.config(text=f"{w_a:.2f} g")
        self.label_b.config(text=f"{w_b:.2f} g")
        self.root.after(200, self.update_weights)

    def update_blobs(self):
        show_preview = True
        if hasattr(self, "var_show_camera_preview"):
            show_preview = bool(self.var_show_camera_preview.get())
        if hasattr(self, "var_show_camera_preview_exp2"):
            show_preview = show_preview or bool(self.var_show_camera_preview_exp2.get())
        if self.experiment_running:
            show_preview = self.preview_enabled_runtime

        if show_preview:
            now = time.time()
            with self.camera_lock:
                if not self.cap.isOpened() and now - self.last_camera_reopen_attempt >= 2.0:
                    self.last_camera_reopen_attempt = now
                    self.cap.release()
                    self.cap = self._open_camera_capture()
                    self.camera_available = self.cap.isOpened()

                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        # Mark unavailable and let next cycle attempt reopen.
                        self.camera_available = False
                        self.cap.release()
                else:
                    ret = False
                    frame = None
            if ret and frame is not None:
                centers = find_marker_centers(frame, self.detector)
                for c in centers:
                    cv2.circle(frame, c, 6, (0, 255, 0), 2)
                cv2.putText(frame, f"Markers: {len(centers)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Backend: {self.camera_backend_name}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                cv2.imshow("Camera Preview", frame)
                cv2.waitKey(1)

        self.root.after(100, self.update_blobs)

    def close_all(self):
        self.shutdown_requested = True
        self.experiment_running = False
        if self.settings_save_after_id is not None:
            try:
                self.root.after_cancel(self.settings_save_after_id)
            except Exception:
                pass
            self.settings_save_after_id = None
        try:
            self._save_calibration_settings()
        except Exception:
            pass
        if self.experiment_thread is not None and self.experiment_thread.is_alive():
            self.experiment_thread.join(timeout=2.0)
        self.scale.close()
        lgpio.gpiochip_close(self.gpio_handle)
        self.motor_a.close()
        self.motor_b.close()
        with self.camera_lock:
            self.cap.release()
        cv2.destroyAllWindows()
        self._close_experiment_logging()


# =================== MAIN ===================
if __name__ == "__main__":
    root = tk.Tk()
    app = WinchUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.close_all(), root.destroy()))
    root.mainloop()
