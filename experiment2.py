"""Experiment 2 open-loop orchestration."""

import time

from core import EXP2_INTERMOVE_MEASUREMENT_BUDGET_S

def _move_open_loop_phase(self, rep, segment_index, phase, phase_total):
    params = self.run_params
    pulses_left = phase_total
    chunk = max(params["move_chunk_pulses"], 1)
    pulses_moved = 0

    while pulses_left > 0 and self.experiment_running:
        step_pulses = min(chunk, pulses_left)
        self._set_motion_state(True)
        self._move_pair_for_phase(step_pulses, phase)
        pulses_moved += step_pulses
        pulses_left -= step_pulses

        measurement = self._collect_settled_samples(
            max_duration_s=EXP2_INTERMOVE_MEASUREMENT_BUDGET_S,
            min_samples_required=1,
        )
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
            and pulses_moved > 0
            and pulses_moved % params["capture_pulses"] == 0
        ):
            capture_ok, centers = self._capture_and_log_blobs()

        self._log_experiment_data_row(
            rep=rep,
            phase=phase,
            pulses_moved=pulses_moved,
            w_a=w_a,
            w_b=w_b,
            centers=centers if capture_ok else [],
        )

        self._set_ui_state(
            progress=f"Experiment 2 - Repetition {rep+1}/{params['reps']} - {phase.title()}",
            exp_a=w_a,
            exp_b=w_b,
            progress_pct=self._calculate_progress_pct_exp2(
                rep=rep,
                segment_index=segment_index,
                segment_pulses=pulses_moved,
                segment_total=phase_total,
            ),
            phase=f"{phase.title()} {segment_index+1}/4",
            rep_text=f"{rep+1} / {params['reps']}",
            pulse_text=f"{pulses_moved} / {phase_total}",
            correction_text="N/A",
        )

    return "ok" if self.experiment_running else "stopped"

def _run_experiment2_thread(self):
    try:
        params = self.run_params
        dwell_sec = params["dwell_ms"] / 1000.0

        phase_plan = [
            ("forward", params["forward_steps"], "A wind / B release"),
            ("return", params["return_steps"], "A release / B wind"),
            ("forward", params["return_steps"], "A wind / B release"),
            ("return", params["forward_steps"], "A release / B wind"),
        ]

        for rep in range(params["reps"]):
            if not self.experiment_running:
                break

            for segment_index, (phase, phase_steps, phase_label) in enumerate(phase_plan):
                if not self.experiment_running:
                    break
                self._set_ui_state(
                    progress=(
                        f"Experiment 2 - Repetition {rep+1}/{params['reps']} - "
                        f"{phase_label} ({phase_steps} pulses)"
                    )
                )
                status = self._move_open_loop_phase(rep, segment_index, phase, phase_steps)
                if status != "ok":
                    self.experiment_running = False
                    self._set_ui_state(progress=self._status_text_for_failure(status), phase="Failed")
                    return
                if segment_index < len(phase_plan) - 1:
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
