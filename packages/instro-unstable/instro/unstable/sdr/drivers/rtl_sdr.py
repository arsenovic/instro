"""RTL-SDR driver wrapper around ``rtlsdr``.

This is intentionally a thin adapter: it keeps the vendor SDK dependency at the
driver boundary and exposes the minimal SDR semantics needed by the higher-level
instro interface. The driver does not own measurement publishing; the ``InstroSDR``
wrapper does that using the repo's shared ``Measurement`` objects.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from instro.unstable.sdr.sdr import IQCapture, SDRDriverBase
from instro.unstable.sdr.types import Direction


class RTLSDR(SDRDriverBase):
    """RTL-SDR dongle. Connection params captured in ``__init__``; USB opens on ``open()``."""

    # librtlsdr moves whole 512-byte USB blocks; a short read makes pyrtlsdr close the device.
    READ_GRANULARITY: ClassVar[int] = 256

    def __init__(self, device_index: int = 0, **kwargs: Any):
        """``device_index`` picks among connected dongles; ``kwargs`` pass through to ``RtlSdr``."""
        self._device_index = device_index
        self._kwargs = kwargs
        self._device: Any = None

    def open(self) -> None:
        from rtlsdr import RtlSdr  # type: ignore[import-untyped]

        if self._device is None:
            self._device = RtlSdr(device_index=self._device_index, **self._kwargs)

    def close(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None

    def set_center_freq(self, frequency_hz: float, *, direction: Direction = Direction.RX, channel: str = "0") -> None:
        self._require_rx(direction, channel).center_freq = float(frequency_hz)

    def get_center_freq(self, *, direction: Direction = Direction.RX, channel: str = "0") -> float:
        return float(self._require_rx(direction, channel).center_freq)

    def set_sample_rate(
        self, sample_rate_hz: float, *, direction: Direction = Direction.RX, channel: str = "0"
    ) -> None:
        self._require_rx(direction, channel).sample_rate = float(sample_rate_hz)

    def get_sample_rate(self, *, direction: Direction = Direction.RX, channel: str = "0") -> float:
        return float(self._require_rx(direction, channel).sample_rate)

    def set_gain(self, gain_db: float, *, direction: Direction = Direction.RX, channel: str = "0") -> None:
        self._require_rx(direction, channel).gain = float(gain_db)

    def get_gain(self, *, direction: Direction = Direction.RX, channel: str = "0") -> float:
        return float(self._require_rx(direction, channel).gain)

    def set_gain_mode(self, automatic: bool, *, direction: Direction = Direction.RX, channel: str = "0") -> None:
        self._require_rx(direction, channel).set_manual_gain_enabled(not automatic)

    def get_gain_range(self, *, direction: Direction = Direction.RX, channel: str = "0") -> tuple[float, float]:
        gains = self._require_rx(direction, channel).valid_gains_db
        return float(min(gains)), float(max(gains))

    def set_bandwidth(self, bandwidth_hz: float, *, direction: Direction = Direction.RX, channel: str = "0") -> None:
        self._require_rx(direction, channel).bandwidth = float(bandwidth_hz)

    def get_bandwidth(self, *, direction: Direction = Direction.RX, channel: str = "0") -> float:
        return float(self._require_rx(direction, channel).bandwidth)

    def set_freq_correction(self, ppm: float, *, direction: Direction = Direction.RX, channel: str = "0") -> None:
        self._require_rx(direction, channel).freq_correction = int(ppm)

    def get_freq_correction(self, *, direction: Direction = Direction.RX, channel: str = "0") -> float:
        return float(self._require_rx(direction, channel).freq_correction)

    def get_num_channels(self, direction: Direction = Direction.RX) -> int:
        return 1 if direction is Direction.RX else 0

    def read_iq(self, n_samples: int, *, direction: Direction = Direction.RX, channel: str = "0") -> IQCapture:
        """Read ``n_samples`` complex IQ pairs from the device."""
        if n_samples <= 0 or n_samples % self.READ_GRANULARITY:
            raise ValueError(f"n_samples must be a positive multiple of {self.READ_GRANULARITY}, got {n_samples}")
        device = self._require_rx(direction, channel)
        samples = np.asarray(device.read_samples(n_samples), dtype=np.complex128)
        # No sample clock of its own, so t0 stays unset and the host anchors the block.
        return IQCapture(
            samples=samples,
            sample_period_ns=1e9 / float(device.sample_rate),
            center_freq_hz=float(device.center_freq),
        )

    def _require_rx(self, direction: Direction, channel: str) -> Any:
        """An RTL-SDR is receive-only with a single signal path."""
        if direction is not Direction.RX or channel != "0":
            raise ValueError(f"RTLSDR has only rx channel '0', got {direction.value} channel '{channel}'")
        return self._require_device()

    def _require_device(self) -> Any:
        # pyrtlsdr closes on a transport error; reading through the stale handle segfaults.
        if self._device is not None and not getattr(self._device, "device_opened", True):
            self._device = None
        if self._device is None:
            raise RuntimeError("RTLSDR driver is not open; call open() first")
        return self._device
