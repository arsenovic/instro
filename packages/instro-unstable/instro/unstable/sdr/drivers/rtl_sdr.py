"""RTL-SDR driver wrapper around ``rtlsdr``.

This is intentionally a thin adapter: it keeps the vendor SDK dependency at the
driver boundary and exposes the minimal SDR semantics needed by the higher-level
instro interface. The driver does not own measurement publishing; the ``InstroSDR``
wrapper does that using the repo's shared ``Measurement`` objects.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from instro.unstable.sdr.sdr import SDRDriverBase


class RTLSDR(SDRDriverBase):
    """RTL-SDR dongle. Connection params captured in ``__init__``; USB opens on ``open()``."""

    # librtlsdr transfers whole 512-byte USB blocks and one IQ sample is 2 bytes; a request
    # that is not a whole number of blocks short-reads, which makes pyrtlsdr close the device.
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

    def set_center_freq(self, frequency_hz: float) -> None:
        self._require_device().center_freq = float(frequency_hz)

    def get_center_freq(self) -> float:
        return float(self._require_device().center_freq)

    def set_sample_rate(self, sample_rate_hz: float) -> None:
        self._require_device().sample_rate = float(sample_rate_hz)

    def get_sample_rate(self) -> float:
        return float(self._require_device().sample_rate)

    def set_gain(self, gain_db: float) -> None:
        self._require_device().gain = float(gain_db)

    def get_gain(self) -> float:
        return float(self._require_device().gain)

    def set_bandwidth(self, bandwidth_hz: float) -> None:
        self._require_device().bandwidth = float(bandwidth_hz)

    def get_bandwidth(self) -> float:
        return float(self._require_device().bandwidth)

    def read_iq(self, n_samples: int) -> np.ndarray:
        """Read ``n_samples`` complex IQ pairs from the device."""
        if n_samples <= 0 or n_samples % self.READ_GRANULARITY:
            raise ValueError(f"n_samples must be a positive multiple of {self.READ_GRANULARITY}, got {n_samples}")
        return np.asarray(self._require_device().read_samples(n_samples), dtype=np.complex128)

    def _require_device(self) -> Any:
        # pyrtlsdr closes the handle itself on a transport error, so a reference we still hold
        # can point at a closed device -- reading through it segfaults inside librtlsdr.
        if self._device is not None and not getattr(self._device, "device_opened", True):
            self._device = None
        if self._device is None:
            raise RuntimeError("RTLSDR driver is not open; call open() first")
        return self._device
