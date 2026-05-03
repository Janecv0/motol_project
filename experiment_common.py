"""Shared line-break helpers for experiment modes."""

from core import DEFAULT_LINE_BREAK_BREACHES, DEFAULT_LINE_BREAK_DROP_G, DEFAULT_LINE_BREAK_DROP_PCT, DEFAULT_LINE_BREAK_WINDOW, MIN_MULTIPLIER_ABS

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
