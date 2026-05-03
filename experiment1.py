"""Experiment 1 control-loop orchestration."""

import time

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
