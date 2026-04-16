# Motol Project: Winch + Scales + Blob Logger

Python application for a seesaw joint-fatigue / bearing test rig controlled from Raspberry Pi.

The project combines:
- `tkinter` GUI
- HX711 dual-channel load-cell reading (`lgpio`)
- dual stepper winch control
- optional camera/blob tracking (`OpenCV`)
- experiment logging to CSV

## Repository Contents
- `test.py`: Main GUI app (calibration, motor control, experiment control, logging)
- `scale_test.py`: Lightweight CLI test for HX711 channels A/B
- `finger_spotter.py`: Separate finger marker tracking/angle logging tool
- `jak_se_tam_prihlasit.txt`: local helper notes for SSH/SCP

## Main Features (`test.py`)
- Live scale readings for two sides of the seesaw
- Independent Scale A / Scale B calibration with persistence
- Manual motor jog controls + quick step buttons
- Experiment mode with robust control loop:
  - move -> settle -> sample -> filter -> adjust
- Experiment 2 mode with open-loop chunked motion:
  - move -> settle -> sample -> log
- Line-break safety stop (sudden drop detection) for both experiment modes
- Optional image capture during experiment
- Optional live camera preview
- Detailed experiment status panel and progress bar
- Structured CSV diagnostics for post-run analysis

---

## Hardware / Wiring Assumptions (Current Code)

### HX711
- `DT`: GPIO `5`
- `SCK`: GPIO `6`
- Single HX711 board using both channels:
  - Channel A and B are read from one board

### Motors
- Motor A:
  - `PUL`: GPIO `17`
  - `DIR`: GPIO `27`
- Motor B:
  - `PUL`: GPIO `23`
  - `DIR`: GPIO `24`

### Camera
- OpenCV camera index `0` (USB camera)

### Important mapping notes
- UI scale mapping is intentionally swapped in code:
  - UI **Scale A** reads HX711 channel **B**
  - UI **Scale B** reads HX711 channel **A**
- Motor B direction is wired/opposed versus Motor A; this is handled in software.

---

## Requirements

Recommended on Raspberry Pi OS:
- Python 3.10+ (project currently also compiles on 3.14)
- `tkinter`
- `lgpio`
- `opencv-python` (or system OpenCV package)
- `numpy` and `matplotlib` (for `finger_spotter.py` and `experiment_visualizer.py`)

Example install (Debian/Raspberry Pi OS packages):
```bash
sudo apt update
sudo apt install -y python3-tk python3-lgpio python3-opencv python3-numpy python3-matplotlib
```

If you prefer pip for OpenCV:
```bash
pip3 install opencv-python
```

---

## Run the Main App
From the project directory:
```bash
python3 test.py
```

On app close, GPIO and camera resources are released cleanly.

## Run the Experiment Visualizer
Visualize saved experiment logs (`experiment_*_data.csv` and `experiment2_*_data.csv`) with:
- time-series plots for tension and selected blob coordinates
- animated blob movement with per-blob color coding and trails

From the project directory:
```bash
python3 experiment_visualizer.py
```

Optional arguments:
```bash
python3 experiment_visualizer.py --experiments-dir /path/to/experiments
python3 experiment_visualizer.py --initial-experiment experiment_YYYYMMDD_HHMMSS
python3 experiment_visualizer.py --initial-experiment experiment2_YYYYMMDD_HHMMSS
```

---

## GUI Walkthrough

## 1) Scale Calibration Tab
This tab now supports independent calibration for each scale.

### Live controls
- `Tare A` / `Tare B`: zero each scale independently (unloaded)
- Live readout in grams for Scale A and Scale B

### Guided known-weight calibration (per scale)
For each scale:
1. Leave scale unloaded, click `Tare`.
2. Place a known weight.
3. Enter `Known Weight (g)`.
4. Click `Calibrate ... from Known Weight`.

The app computes:
- `multiplier = (loaded_raw - tare_raw) / known_weight_g`

### Manual multiplier override (per scale)
- Edit `Scale A Multiplier` or `Scale B Multiplier`
- Click `Apply Manual A/B`

### Multiplier sign
- Multiplier can be positive or negative.
- Sign depends on pull/load direction and sensor orientation.
- Only finite, non-zero multipliers are accepted.

### Persistence
Calibration is auto-saved to:
- `scale_calibration.json` in current working directory

Schema:
```json
{
  "version": 1,
  "cal_a": 400.0,
  "cal_b": 400.0
}
```

Startup behavior:
- Valid file: loaded
- Missing file: defaults (`400`, `400`)
- Invalid file: defaults + warning

Calibration actions are blocked while experiment is running.

---

## 2) Blob Setup Tab
- Tune blob detection (`minArea`, `maxArea`)
- Live camera preview window with marker circles
- Used by experiment capture pipeline

---

## 3) Motor Control Tab
- Quick preset moves for Motor A / Motor B
- Custom pulse count with direction selection
- `Move A`, `Move B`, `Move Both`

Internal motor move includes simple acceleration/deceleration ramp for smoother motion.

---

## 4) Experiment Tab
This is the robust control mode for oscillating the seesaw under load.

### Experiment parameters
- `Target Tension A (g)`, `Target Tension B (g)`
- `Movement Amplitude (pulses)`
- `Dwell Time (ms)`
- `Repetitions`
- `Image Capture Every (pulses)` (only used when capture enabled)
- `Adjustment Step (pulses)`
- `Tolerance (g)`
- `Measurement Delay (ms)` (settling delay after movement)
- `Measurement Samples`
- `Sample Interval (ms)`
- `Stabilization Timeout (s)`
- `Max Correction Cycles`
- `Move Chunk (pulses)` (smaller chunk = smoother motion/corrections)
- `Line-break Drop (g)`
- `Line-break Drop (%)`
- `Safety Baseline Window (samples)`
- `Safety Consecutive Breaches`
- `Enable Image Capture` checkbox
- `Show Camera Preview` checkbox

### Capture behavior
- If capture is enabled and camera is unavailable:
  - app asks whether to continue without capture

### Status panel
- Current A/B filtered values
- Progress text + percent bar
- Phase, repetition, pulses, correction count, capture status

---

## Control Strategy in Experiment Mode

High-level loop:
1. Stabilize tension to targets.
2. Forward phase movement with periodic correction.
3. Return phase movement with periodic correction.
4. Repeat.

Each control step follows:
1. Move motors.
2. Wait settling delay.
3. Collect N samples.
4. Filter readings.
5. Compute correction.
6. Apply correction if needed.

### Filtering used
- Median of sample batch (outlier rejection)
- EMA smoothing (`alpha = 0.3`) for control stability

### Scale reads during motion
- Reads are blocked while motion/settling is in progress.
- UI shows last valid settled measurement during movement.

---

## 5) Experiment 2 Tab
Open-loop mode for fixed chunked motion without tension correction.

### Experiment 2 parameters
- `Movement Amplitude (pulses)`
- `Motor Step Chunk (pulses)`
- `Motor A Move Scale`
- `Motor B Move Scale`
- `Dwell Time (ms)`
- `Repetitions`
- `Image Capture Every (pulses)`
- `Measurement Delay (ms)`
- `Measurement Samples`
- `Sample Interval (ms)`
- `Line-break Drop (g)`
- `Line-break Drop (%)`
- `Safety Baseline Window (samples)`
- `Safety Consecutive Breaches`
- `Enable Image Capture` checkbox
- `Show Camera Preview` checkbox

### Experiment 2 behavior
- Runs forward and return phases per repetition
- Logs filtered weight after every movement chunk
- Includes blob coordinates when capture is triggered; otherwise blob columns remain blank
- Stops immediately when line-break safety condition is confirmed

---

## Logging

Main experiment log files:
- `experiment_YYYYMMDD_HHMMSS_data.csv` / `experiment_YYYYMMDD_HHMMSS_setup.csv`
- `experiment2_YYYYMMDD_HHMMSS_data.csv` / `experiment2_YYYYMMDD_HHMMSS_setup.csv`

Columns include:
- repetition/phase/pulses
- tensions
- blob coordinates (`blob0_x`, `blob0_y`, ...)

Calibration file:
- `scale_calibration.json`

Finger tracker output:
- `finger_data.csv` (from `finger_spotter.py`)

---

## Support Script: `scale_test.py`
Use for fast HX711 sanity checks without GUI.

Example:
```bash
python3 scale_test.py --dt 5 --sck 6 --cal-a 400 --cal-b 400
```

Useful options:
- `--interval`
- `--tare-samples`
- `--ready-timeout`

---

## Support Script: `finger_spotter.py`
Standalone marker tracker/angle logger.

Run:
```bash
python3 finger_spotter.py
```

Controls:
- `c`: calibrate marker order on straight finger
- `q`: quit

Outputs:
- live overlay
- `finger_data.csv`
- final matplotlib angle plot

---

## Raspberry Pi SSH / SCP

Typical workflow:
```bash
ssh screw@raspberrypi.local
```

Copy file to Pi user home:
```bash
scp /path/to/local/file.py screw@raspberrypi.local:/home/screw/
```

If host discovery via `.local` fails, use direct IP.

---

## Troubleshooting

## Experiment stops with `timeout` or `max_corrections`
- Increase `Stabilization Timeout (s)`
- Increase `Max Correction Cycles`
- Increase `Measurement Delay (ms)` if mechanical settling is slow
- Reduce `Move Chunk (pulses)` for smoother motion
- Verify targets are physically reachable

## Scale reads near zero or unstable
- Re-check wiring and load cell polarity
- Confirm correct channel mapping (A/B share one HX711 board)
- Re-tare unloaded scale
- Re-calibrate with known weight
- If sign is inverted, use negative multiplier

## Camera not connected
- Disable `Enable Image Capture` or continue without capture when prompted
- Optionally disable preview too

## `scp: dest open ... Failure`
- Destination path/user mismatch is common
- Use writable home dir of actual SSH user, e.g. `/home/screw/`

## GPIO access issues
- Ensure running on Pi with `lgpio` installed
- Verify user permissions for GPIO device access

---

## Safety / Operation Notes
- Keep clear of moving strings/pulleys during runs.
- Start with low amplitude and small correction steps.
- Verify emergency stop behavior (`Stop` button) before high-load tests.
- Re-run tare/calibration if rig geometry or load path changes.

---

## Suggested Baseline Parameters (First Bring-Up)
- Target A/B: low and equal (e.g. 30-60 g)
- Move amplitude: 10-20 pulses
- Move chunk: 1-2 pulses
- Adjustment step: 1 pulse
- Tolerance: 2-5 g
- Measurement delay: 150-300 ms
- Measurement samples: 7
- Sample interval: 20 ms
- Stabilization timeout: 10-20 s
- Max correction cycles: 80+

Tune upward once behavior is stable.
