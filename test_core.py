import unittest

from core import (
    calculate_progress_pct,
    calculate_progress_pct_exp2,
    compute_correction,
    compute_init_correction,
    is_valid_multiplier,
    step_delay_profile,
    validate_experiment2_params,
    validate_experiment_params,
)


class CoreHelpersTest(unittest.TestCase):
    def test_compute_correction(self):
        self.assertEqual(compute_correction(5, 2, 1), 1)
        self.assertEqual(compute_correction(-5, 2, 1), -1)
        self.assertEqual(compute_correction(1, 2, 1), 0)

    def test_compute_init_correction(self):
        self.assertEqual(compute_init_correction(1, 2, 1, 0.2), 0)
        self.assertEqual(compute_init_correction(10, 2, 1, 0.2), 2)
        self.assertEqual(compute_init_correction(-10, 2, 1, 0.2), -2)

    def test_progress_helpers(self):
        self.assertGreater(calculate_progress_pct(0, "forward", 5, 10, 5), 0)
        self.assertEqual(calculate_progress_pct(0, "done", reps_total=5), 100.0)
        self.assertGreater(calculate_progress_pct_exp2(0, 0, 5, 10, 5), 0)

    def test_step_delay_profile(self):
        self.assertGreater(step_delay_profile(0, 10), step_delay_profile(5, 10))
        self.assertGreater(step_delay_profile(9, 10), step_delay_profile(5, 10))

    def test_multiplier_validation(self):
        self.assertTrue(is_valid_multiplier(1.0))
        self.assertFalse(is_valid_multiplier(0.0))

    def test_validation_helpers(self):
        base_exp = {
            "move_amp": 1,
            "reps": 1,
            "init_adj_step": 1,
            "init_kp": 0.0,
            "adj_step": 1,
            "tolerance": 0,
            "dwell_ms": 0,
            "measurement_delay_ms": 0,
            "measurement_samples": 1,
            "sample_interval_ms": 0,
            "stabilization_timeout_s": 1,
            "max_correction_cycles": 1,
            "move_chunk_pulses": 1,
            "line_break_drop_g": 1,
            "line_break_drop_pct": 1,
            "line_break_window_samples": 1,
            "line_break_consecutive_breaches": 1,
            "enable_capture": False,
            "capture_pulses": 1,
        }
        self.assertIsNone(validate_experiment_params(base_exp))

        base_exp2 = {
            "forward_steps": 1,
            "return_steps": 1,
            "reps": 1,
            "dwell_ms": 0,
            "move_chunk_pulses": 1,
            "measurement_delay_ms": 0,
            "measurement_samples": 1,
            "sample_interval_ms": 0,
            "line_break_drop_g": 1,
            "line_break_drop_pct": 1,
            "line_break_window_samples": 1,
            "line_break_consecutive_breaches": 1,
            "enable_capture": False,
            "capture_pulses": 1,
        }
        self.assertIsNone(validate_experiment2_params(base_exp2))


if __name__ == "__main__":
    unittest.main()
