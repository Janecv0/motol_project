"""Hardware adapters for HX711 and the winch motors."""

from __future__ import annotations

import time

import lgpio


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
            self._read_once_set_next(channel)
        return self._read_once_set_next(channel)

    def read_raw_pair(self, first_channel, second_channel):
        first_channel = first_channel.upper()
        second_channel = second_channel.upper()
        if first_channel not in self.PULSES_FOR_NEXT or second_channel not in self.PULSES_FOR_NEXT:
            raise ValueError("channels must be 'A' or 'B'")

        if first_channel == second_channel:
            raw = self.read_raw(first_channel)
            return raw, raw

        if self.current_channel != first_channel:
            self._read_once_set_next(first_channel)

        raw_first = self._read_once_set_next(second_channel)
        raw_second = self._read_once_set_next(first_channel)
        return raw_first, raw_second

    def tare(self, channel, samples=15):
        return sum(self.read_raw(channel) for _ in range(samples)) / samples

    def get_weight(self, channel, tare_value, calibration):
        raw = self.read_raw(channel)
        return (raw - tare_value) / calibration

    def close(self):
        if self.own_handle:
            lgpio.gpiochip_close(self.h)


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
