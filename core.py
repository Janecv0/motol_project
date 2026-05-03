"""Pure helpers and shared constants for the motol app."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

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
EXP2_INTERMOVE_MEASUREMENT_BUDGET_S = 0.48


@dataclass(frozen=True)
class ExperimentTiming:
    measurement_delay_ms: int
    measurement_samples: int
    sample_interval_ms: int


@dataclass(frozen=True)
class ExperimentMotion:
    move_chunk_pulses: int
    dwell_ms: int
    reps: int


def calculate_progress_pct(rep, phase, phase_pulses=0, phase_total=1, reps_total=1):
    reps_total = max(int(reps_total), 1)
    total_motion_segments = max(reps_total * 2, 1)
    rep = max(0, min(int(rep), reps_total - 1))
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


def calculate_progress_pct_exp2(rep, segment_index, segment_pulses=0, segment_total=1, reps_total=1):
    reps_total = max(int(reps_total), 1)
    segments_per_rep = 4
    total_segments = max(reps_total * segments_per_rep, 1)
    rep = max(0, min(int(rep), reps_total - 1))
    segment_index = max(0, min(int(segment_index), segments_per_rep - 1))
    segment_total = max(int(segment_total), 1)
    segment_pulses = max(0, min(int(segment_pulses), segment_total))

    global_segment_index = rep * segments_per_rep + segment_index
    fraction = (global_segment_index + (segment_pulses / segment_total)) / total_segments
    return max(0.0, min(100.0, fraction * 100.0))


def compute_correction(error, tolerance, adj_step):
    if error > tolerance:
        return adj_step
    if error < -tolerance:
        return -adj_step
    return 0


def compute_init_correction(error, tolerance, min_step, init_kp):
    if abs(error) <= tolerance:
        return 0

    proportional_step = int(round(abs(error) * init_kp))
    pulses = max(int(min_step), proportional_step)
    pulses = max(1, pulses)

    if error > tolerance:
        return pulses
    return -pulses


def step_delay_profile(pulse_index, total_pulses, base_delay=0.0001, ramp_steps=20, ramp_factor=2.5):
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


def validate_experiment_params(params):
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


def validate_experiment2_params(params):
    if params["forward_steps"] <= 0 or params["return_steps"] <= 0 or params["reps"] <= 0:
        return "Forward steps, return steps, and repetitions must be > 0"
    if params["dwell_ms"] < 0:
        return "Dwell time must be >= 0"
    if params["move_chunk_pulses"] <= 0:
        return "Motor step chunk must be > 0"
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


def is_valid_multiplier(value):
    return isfinite(value) and abs(value) > MIN_MULTIPLIER_ABS
