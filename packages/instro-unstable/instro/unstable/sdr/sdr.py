"""SDR instrument API.

This submodule defines the generic SDR contract used throughout the unstable
instrument layer and a thin wrapper around a vendor driver. The design follows the
repo's pattern: the low-level driver owns the hardware transport and device
lifecycle, while the high-level ``InstroSDR`` object publishes measurements and
commands using the shared core measurement model.

The data model intentionally batches IQ samples into a single ``Measurement`` per
acquisition block instead of creating one measurement object per sample. The
``Measurement`` type in ``instro.lib.types`` is meant for a block with a common
shared timebase, not for millions of individual sample-level events.
"""

from __future__ import annotations

import abc
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from instro.lib import Command, Instrument, Measurement
from instro.lib.instrument import publish_command, publish_measurement

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class IQCapture:
    """One block of IQ samples with the timebase and tuner state they were taken under."""

    samples: np.ndarray
    sample_period_ns: float
    center_freq_hz: float
    t0_ns: int | None = None


class SDRDriverBase(abc.ABC):
    """Vendor SDR driver contract. Concrete drivers own their transport and lifecycle."""

    @abc.abstractmethod
    def open(self) -> None:
        """Open the underlying transport or SDR handle."""
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Close the underlying transport or SDR handle."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_center_freq(self, frequency_hz: float) -> None:
        """Set the RF center frequency in Hz."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_center_freq(self) -> float:
        """Get the RF center frequency in Hz."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_sample_rate(self, sample_rate_hz: float) -> None:
        """Set the sample rate in samples per second."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_sample_rate(self) -> float:
        """Get the sample rate in samples per second."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_gain(self, gain_db: float) -> None:
        """Set the gain in dB."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_gain(self) -> float:
        """Get the gain in dB."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_bandwidth(self, bandwidth_hz: float) -> None:
        """Set the IF or filter bandwidth in Hz."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_bandwidth(self) -> float:
        """Get the IF or filter bandwidth in Hz."""
        raise NotImplementedError

    @abc.abstractmethod
    def read_iq(self, n_samples: int) -> IQCapture:
        """Read a block of complex IQ samples together with the timebase they were taken on."""
        raise NotImplementedError


class InstroSDR(Instrument):
    """Higher-level SDR wrapper that publishes buffered IQ measurements."""

    def __init__(self, name: str, driver: SDRDriverBase, **kwargs: Any):
        super().__init__(name, **kwargs)
        self._driver = driver
        self._resource_lock = threading.Lock()
        self._last_iq_timestamp: int | None = None
        self._sample_period_warning_issued = False

    def open(self) -> None:
        """Open the underlying driver."""
        self._driver.open()

    def close(self) -> None:
        """Close the underlying driver and stop the daemon if present."""
        super().close()
        self._driver.close()

    @property
    def driver(self) -> SDRDriverBase:
        """Return the underlying hardware driver."""
        return self._driver

    def _iq_timestamps(self, capture: IQCapture, t_read_ns: int) -> list[int]:
        """Nanosecond timestamps for one capture, spaced at the sample period the device reported."""
        period_ns = capture.sample_period_ns
        if period_ns <= 0:
            raise ValueError(f"driver reported a non-positive sample period: {period_ns}")
        if period_ns < 1.0 and not self._sample_period_warning_issued:
            self._sample_period_warning_issued = True
            logger.warning(
                "Sample period %s ns is shorter than the integer nanosecond timestamps can resolve; "
                "samples in each block will share timestamps.",
                period_ns,
            )

        # Round per index, not a pre-rounded period: 417 ns for 416.667 is 800 ppm of drift.
        offsets = np.round(np.arange(len(capture.samples)) * period_ns).astype(np.int64)

        if capture.t0_ns is not None:
            t0 = capture.t0_ns
        else:
            # Backstamp: the read returned after the samples were taken. Buffered samples
            # continue the previous timeline; a wider gap is a real dropout.
            t0 = t_read_ns - int(offsets[-1])
            if self._last_iq_timestamp is not None and t0 <= self._last_iq_timestamp:
                t0 = self._last_iq_timestamp + round(period_ns)

        timestamps: list[int] = (t0 + offsets).tolist()
        self._last_iq_timestamp = timestamps[-1]
        return timestamps

    def _read_iq_block(self, n_samples: int) -> tuple[IQCapture, list[int]] | None:
        """Acquire one capture and resolve its timestamps. Publishes nothing."""
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        with self._resource_lock:
            capture = self._driver.read_iq(n_samples)
            t_read_ns = time.time_ns()
            if capture.samples.size == 0:
                return None
            return capture, self._iq_timestamps(capture, t_read_ns)

    @publish_measurement
    def measure_iq(self, n_samples: int = 1024, **kwargs: Any) -> Measurement | None:
        """Return a ``Measurement`` containing one IQ buffer block.

        The measurement contains real and imaginary channels as arrays with a common
        timestamp vector. This keeps the payload structured and efficient while
        avoiding one published object per individual sample.
        """
        block = self._read_iq_block(n_samples)
        if block is None:
            return None
        capture, timestamps = block

        return Measurement(
            channel_data={
                f"{self.name}.i": capture.samples.real.tolist(),
                f"{self.name}.q": capture.samples.imag.tolist(),
            },
            timestamps=timestamps,
            tags={**self.default_tags, **kwargs},
        )

    @staticmethod
    def _psd_from_capture(capture: IQCapture) -> tuple[np.ndarray, np.ndarray]:
        """Hann-windowed power spectral density: absolute frequencies and linear PSD."""
        sample_rate_hz = 1e9 / capture.sample_period_ns
        window = np.hanning(len(capture.samples))
        spectrum = np.fft.fftshift(np.fft.fft(capture.samples * window))
        psd = np.abs(spectrum) ** 2 / (sample_rate_hz * np.sum(window**2))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(capture.samples), d=1.0 / sample_rate_hz))
        return freqs + capture.center_freq_hz, psd

    @staticmethod
    def _occupied_bandwidth(freqs: np.ndarray, psd: np.ndarray, fraction: float = 0.99) -> float:
        """Width of the band holding ``fraction`` of total power, split evenly across both tails."""
        total = float(psd.sum())
        if total <= 0:
            return 0.0
        cumulative = np.cumsum(psd) / total
        tail = (1.0 - fraction) / 2.0
        low = int(np.searchsorted(cumulative, tail))
        high = min(int(np.searchsorted(cumulative, 1.0 - tail)), len(freqs) - 1)
        return float(freqs[high] - freqs[low])

    def compute_psd(self, n_samples: int = 1024) -> tuple[np.ndarray, np.ndarray] | None:
        """Acquire a block and return its ``(frequencies_hz, power_db)`` spectrum. Publishes nothing."""
        block = self._read_iq_block(n_samples)
        if block is None:
            return None
        freqs, psd = self._psd_from_capture(block[0])
        return freqs, 10.0 * np.log10(np.maximum(psd, np.finfo(float).tiny))

    @publish_measurement
    def measure_spectrum(self, n_samples: int = 1024, **kwargs: Any) -> Measurement | None:
        """Publish scalar features of the block's power spectrum; use ``compute_psd`` for the array."""
        block = self._read_iq_block(n_samples)
        if block is None:
            return None
        capture, timestamps = block

        freqs, psd = self._psd_from_capture(capture)
        floor = np.finfo(float).tiny
        peak = int(np.argmax(psd))

        return Measurement(
            channel_data={
                f"{self.name}.spectrum.peak_power_db": [float(10.0 * np.log10(max(psd[peak], floor)))],
                f"{self.name}.spectrum.peak_freq_hz": [float(freqs[peak])],
                f"{self.name}.spectrum.mean_power_db": [float(10.0 * np.log10(max(psd.mean(), floor)))],
                f"{self.name}.spectrum.occupied_bw_hz": [self._occupied_bandwidth(freqs, psd)],
            },
            timestamps=[timestamps[-1]],
            tags={**self.default_tags, **kwargs},
        )

    @publish_measurement
    def _execute_measurement(self, driver_method: Callable[[], float], descriptor: str, **kwargs: Any) -> Measurement:
        """Execute a no-argument driver read and return a Measurement for the value."""
        with self._resource_lock:
            val = driver_method()
            timestamp = time.time_ns()
        return self._package_measurement(descriptor, val, timestamp, **kwargs)

    @publish_command
    def _execute_command(
        self, driver_method: Callable[[float], None], value: float, descriptor: str, **kwargs: Any
    ) -> Command:
        """Execute a single-argument driver write and return a Command for the written value."""
        with self._resource_lock:
            driver_method(value)
            timestamp = time.time_ns()
        return self._package_command(f"{descriptor}.cmd", value, timestamp, **kwargs)

    def set_center_freq(self, frequency_hz: float, **kwargs: Any) -> Command:
        """Set RF center frequency and publish the command."""
        return self._execute_command(self._driver.set_center_freq, float(frequency_hz), "center_freq", **kwargs)

    def get_center_freq(self, **kwargs: Any) -> Measurement:
        """Query the current RF center frequency in Hz."""
        return self._execute_measurement(self._driver.get_center_freq, "center_freq", **kwargs)

    def set_sample_rate(self, sample_rate_hz: float, **kwargs: Any) -> Command:
        """Set the sample rate and publish the command."""
        return self._execute_command(self._driver.set_sample_rate, float(sample_rate_hz), "sample_rate", **kwargs)

    def get_sample_rate(self, **kwargs: Any) -> Measurement:
        """Query the current sample rate in samples per second."""
        return self._execute_measurement(self._driver.get_sample_rate, "sample_rate", **kwargs)

    def set_gain(self, gain_db: float, **kwargs: Any) -> Command:
        """Set the gain in dB and publish the command."""
        return self._execute_command(self._driver.set_gain, float(gain_db), "gain", **kwargs)

    def get_gain(self, **kwargs: Any) -> Measurement:
        """Query the current gain in dB."""
        return self._execute_measurement(self._driver.get_gain, "gain", **kwargs)

    def set_bandwidth(self, bandwidth_hz: float, **kwargs: Any) -> Command:
        """Set the bandwidth in Hz and publish the command."""
        return self._execute_command(self._driver.set_bandwidth, float(bandwidth_hz), "bandwidth", **kwargs)

    def get_bandwidth(self, **kwargs: Any) -> Measurement:
        """Query the current IF or filter bandwidth in Hz."""
        return self._execute_measurement(self._driver.get_bandwidth, "bandwidth", **kwargs)
