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
from enum import Enum
from typing import Any, Callable

import numpy as np

from instro.lib import Command, Instrument, Measurement
from instro.lib.instrument import publish_command, publish_measurement
from instro.unstable.sdr.types import Direction

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class IQCapture:
    """One block of IQ samples with the timebase and tuner state they were taken under."""

    samples: np.ndarray
    sample_period_ns: float
    center_freq_hz: float
    t0_ns: int | None = None


class SDRDriverBase(abc.ABC):
    """Vendor SDR driver contract.

    Required methods are abstract: every radio can tune, set a rate, and hand back
    samples. Everything else raises ``NotImplementedError`` by default, and a driver
    overrides only what its hardware actually supports.

    ``direction`` and ``channel`` address one signal path. A receive-only, single-path
    radio accepts the defaults and rejects anything else; drivers validate their own
    arguments rather than relying on a shared helper.
    """

    # --- Required ---

    @abc.abstractmethod
    def open(self) -> None:
        """Open the underlying transport or SDR handle."""
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Close the underlying transport or SDR handle."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_center_freq(self, frequency_hz: float, *, direction: Direction = Direction.RX, channel: str = "0") -> None:
        """Set the RF center frequency in Hz."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_center_freq(self, *, direction: Direction = Direction.RX, channel: str = "0") -> float:
        """Get the RF center frequency in Hz."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_sample_rate(
        self, sample_rate_hz: float, *, direction: Direction = Direction.RX, channel: str = "0"
    ) -> None:
        """Set the sample rate in samples per second."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_sample_rate(self, *, direction: Direction = Direction.RX, channel: str = "0") -> float:
        """Get the sample rate in samples per second."""
        raise NotImplementedError

    @abc.abstractmethod
    def read_iq(self, n_samples: int, *, direction: Direction = Direction.RX, channel: str = "0") -> IQCapture:
        """Read a block of complex IQ samples together with the timebase they were taken on."""
        raise NotImplementedError

    # --- Optional: gain ---

    def set_gain(self, gain_db: float, *, direction: Direction = Direction.RX, channel: str = "0") -> None:
        """Set the overall gain in dB. Override if the radio exposes a single gain figure."""
        raise NotImplementedError("Gain control has not been implemented for this driver")

    def get_gain(self, *, direction: Direction = Direction.RX, channel: str = "0") -> float:
        """Get the overall gain in dB."""
        raise NotImplementedError("Gain control has not been implemented for this driver")

    def set_gain_mode(self, automatic: bool, *, direction: Direction = Direction.RX, channel: str = "0") -> None:
        """Hand gain to the radio's AGC (``True``) or take manual control (``False``)."""
        raise NotImplementedError("Gain mode has not been implemented for this driver")

    def get_gain_mode(self, *, direction: Direction = Direction.RX, channel: str = "0") -> bool:
        """Report whether the AGC is in control. Many radios cannot read this back."""
        raise NotImplementedError("Gain mode readback has not been implemented for this driver")

    def get_gain_range(self, *, direction: Direction = Direction.RX, channel: str = "0") -> tuple[float, float]:
        """Lowest and highest settable gain in dB."""
        raise NotImplementedError("Gain range has not been implemented for this driver")

    # --- Optional: front end ---

    def set_bandwidth(self, bandwidth_hz: float, *, direction: Direction = Direction.RX, channel: str = "0") -> None:
        """Set the IF or filter bandwidth in Hz. Some radios tie this to the sample rate."""
        raise NotImplementedError("Bandwidth control has not been implemented for this driver")

    def get_bandwidth(self, *, direction: Direction = Direction.RX, channel: str = "0") -> float:
        """Get the IF or filter bandwidth in Hz."""
        raise NotImplementedError("Bandwidth control has not been implemented for this driver")

    def set_freq_correction(self, ppm: float, *, direction: Direction = Direction.RX, channel: str = "0") -> None:
        """Correct the reference oscillator error in parts per million."""
        raise NotImplementedError("Frequency correction has not been implemented for this driver")

    def get_freq_correction(self, *, direction: Direction = Direction.RX, channel: str = "0") -> float:
        """Get the reference oscillator correction in parts per million."""
        raise NotImplementedError("Frequency correction has not been implemented for this driver")

    def list_antennas(self, *, direction: Direction = Direction.RX, channel: str = "0") -> list[str]:
        """Names of the selectable antenna ports."""
        raise NotImplementedError("Antenna selection has not been implemented for this driver")

    def set_antenna(self, antenna: str, *, direction: Direction = Direction.RX, channel: str = "0") -> None:
        """Select an antenna port by name."""
        raise NotImplementedError("Antenna selection has not been implemented for this driver")

    def get_antenna(self, *, direction: Direction = Direction.RX, channel: str = "0") -> str:
        """Name of the selected antenna port."""
        raise NotImplementedError("Antenna selection has not been implemented for this driver")

    # --- Optional: capability discovery ---

    def get_num_channels(self, direction: Direction = Direction.RX) -> int:
        """Number of signal paths in ``direction``. Zero means the radio cannot do it at all."""
        raise NotImplementedError("Channel counts have not been implemented for this driver")

    def get_frequency_range(self, *, direction: Direction = Direction.RX, channel: str = "0") -> tuple[float, float]:
        """Lowest and highest tunable center frequency in Hz."""
        raise NotImplementedError("Frequency range has not been implemented for this driver")

    def get_sample_rate_range(self, *, direction: Direction = Direction.RX, channel: str = "0") -> tuple[float, float]:
        """Lowest and highest settable sample rate in samples per second."""
        raise NotImplementedError("Sample rate range has not been implemented for this driver")


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

    def _read_iq_block(self, n_samples: int, direction: Direction, channel: str) -> tuple[IQCapture, list[int]] | None:
        """Acquire one capture and resolve its timestamps. Publishes nothing."""
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        with self._resource_lock:
            capture = self._driver.read_iq(n_samples, direction=direction, channel=channel)
            t_read_ns = time.time_ns()
            if capture.samples.size == 0:
                return None
            return capture, self._iq_timestamps(capture, t_read_ns)

    @publish_measurement
    def measure_iq(
        self, n_samples: int = 1024, *, direction: Direction = Direction.RX, channel: str = "0", **kwargs: Any
    ) -> Measurement | None:
        """Return a ``Measurement`` containing one IQ buffer block.

        The measurement contains real and imaginary channels as arrays with a common
        timestamp vector. This keeps the payload structured and efficient while
        avoiding one published object per individual sample.
        """
        block = self._read_iq_block(n_samples, direction, channel)
        if block is None:
            return None
        capture, timestamps = block
        path = self._path(direction, channel)

        return Measurement(
            channel_data={
                f"{self.name}.{path}.i": capture.samples.real.tolist(),
                f"{self.name}.{path}.q": capture.samples.imag.tolist(),
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

    def compute_psd(
        self, n_samples: int = 1024, *, direction: Direction = Direction.RX, channel: str = "0"
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Acquire a block and return its ``(frequencies_hz, power_db)`` spectrum. Publishes nothing."""
        block = self._read_iq_block(n_samples, direction, channel)
        if block is None:
            return None
        freqs, psd = self._psd_from_capture(block[0])
        return freqs, 10.0 * np.log10(np.maximum(psd, np.finfo(float).tiny))

    @publish_measurement
    def measure_spectrum(
        self, n_samples: int = 1024, *, direction: Direction = Direction.RX, channel: str = "0", **kwargs: Any
    ) -> Measurement | None:
        """Publish scalar features of the block's power spectrum; use ``compute_psd`` for the array."""
        block = self._read_iq_block(n_samples, direction, channel)
        if block is None:
            return None
        capture, timestamps = block
        path = self._path(direction, channel)

        freqs, psd = self._psd_from_capture(capture)
        floor = np.finfo(float).tiny
        peak = int(np.argmax(psd))

        return Measurement(
            channel_data={
                f"{self.name}.{path}.spectrum.peak_power_db": [float(10.0 * np.log10(max(psd[peak], floor)))],
                f"{self.name}.{path}.spectrum.peak_freq_hz": [float(freqs[peak])],
                f"{self.name}.{path}.spectrum.mean_power_db": [float(10.0 * np.log10(max(psd.mean(), floor)))],
                f"{self.name}.{path}.spectrum.occupied_bw_hz": [self._occupied_bandwidth(freqs, psd)],
            },
            timestamps=[timestamps[-1]],
            tags={**self.default_tags, **kwargs},
        )

    @staticmethod
    def _path(direction: Direction, channel: str) -> str:
        """Descriptor prefix naming one signal path, e.g. ``rx0``."""
        return f"{direction.value}{channel}"

    @publish_measurement
    def _execute_measurement(
        self,
        driver_method: Callable[..., Any],
        descriptor: str,
        direction: Direction,
        channel: str,
        **kwargs: Any,
    ) -> Measurement:
        """Execute a driver read and return a Measurement for the value."""
        with self._resource_lock:
            val = driver_method(direction=direction, channel=channel)
            timestamp = time.time_ns()
        val = val.value if isinstance(val, Enum) else val
        return self._package_measurement(f"{self._path(direction, channel)}.{descriptor}", val, timestamp, **kwargs)

    @publish_command
    def _execute_command(
        self,
        driver_method: Callable[..., None],
        value: float | bool | str,
        descriptor: str,
        direction: Direction,
        channel: str,
        **kwargs: Any,
    ) -> Command:
        """Execute a driver write and return a Command for the written value."""
        with self._resource_lock:
            driver_method(value, direction=direction, channel=channel)
            timestamp = time.time_ns()
        return self._package_command(f"{self._path(direction, channel)}.{descriptor}.cmd", value, timestamp, **kwargs)

    def set_center_freq(
        self, frequency_hz: float, *, direction: Direction = Direction.RX, channel: str = "0", **kwargs: Any
    ) -> Command:
        """Set RF center frequency and publish the command."""
        return self._execute_command(
            self._driver.set_center_freq, float(frequency_hz), "center_freq", direction, channel, **kwargs
        )

    def get_center_freq(self, *, direction: Direction = Direction.RX, channel: str = "0", **kwargs: Any) -> Measurement:
        """Query the current RF center frequency in Hz."""
        return self._execute_measurement(self._driver.get_center_freq, "center_freq", direction, channel, **kwargs)

    def set_sample_rate(
        self, sample_rate_hz: float, *, direction: Direction = Direction.RX, channel: str = "0", **kwargs: Any
    ) -> Command:
        """Set the sample rate and publish the command."""
        return self._execute_command(
            self._driver.set_sample_rate, float(sample_rate_hz), "sample_rate", direction, channel, **kwargs
        )

    def get_sample_rate(self, *, direction: Direction = Direction.RX, channel: str = "0", **kwargs: Any) -> Measurement:
        """Query the current sample rate in samples per second."""
        return self._execute_measurement(self._driver.get_sample_rate, "sample_rate", direction, channel, **kwargs)

    def set_gain(
        self, gain_db: float, *, direction: Direction = Direction.RX, channel: str = "0", **kwargs: Any
    ) -> Command:
        """Set the gain in dB and publish the command."""
        return self._execute_command(self._driver.set_gain, float(gain_db), "gain", direction, channel, **kwargs)

    def get_gain(self, *, direction: Direction = Direction.RX, channel: str = "0", **kwargs: Any) -> Measurement:
        """Query the current gain in dB."""
        return self._execute_measurement(self._driver.get_gain, "gain", direction, channel, **kwargs)

    def set_gain_mode(
        self, automatic: bool, *, direction: Direction = Direction.RX, channel: str = "0", **kwargs: Any
    ) -> Command:
        """Hand gain to the radio's AGC or take manual control, and publish the command."""
        return self._execute_command(
            self._driver.set_gain_mode, bool(automatic), "gain_mode", direction, channel, **kwargs
        )

    def get_gain_mode(self, *, direction: Direction = Direction.RX, channel: str = "0", **kwargs: Any) -> Measurement:
        """Query whether the AGC is in control."""
        return self._execute_measurement(self._driver.get_gain_mode, "gain_mode", direction, channel, **kwargs)

    def set_bandwidth(
        self, bandwidth_hz: float, *, direction: Direction = Direction.RX, channel: str = "0", **kwargs: Any
    ) -> Command:
        """Set the bandwidth in Hz and publish the command."""
        return self._execute_command(
            self._driver.set_bandwidth, float(bandwidth_hz), "bandwidth", direction, channel, **kwargs
        )

    def get_bandwidth(self, *, direction: Direction = Direction.RX, channel: str = "0", **kwargs: Any) -> Measurement:
        """Query the current IF or filter bandwidth in Hz."""
        return self._execute_measurement(self._driver.get_bandwidth, "bandwidth", direction, channel, **kwargs)

    def set_freq_correction(
        self, ppm: float, *, direction: Direction = Direction.RX, channel: str = "0", **kwargs: Any
    ) -> Command:
        """Set the reference oscillator correction in ppm and publish the command."""
        return self._execute_command(
            self._driver.set_freq_correction, float(ppm), "freq_correction", direction, channel, **kwargs
        )

    def get_freq_correction(
        self, *, direction: Direction = Direction.RX, channel: str = "0", **kwargs: Any
    ) -> Measurement:
        """Query the reference oscillator correction in ppm."""
        return self._execute_measurement(
            self._driver.get_freq_correction, "freq_correction", direction, channel, **kwargs
        )

    def set_antenna(
        self, antenna: str, *, direction: Direction = Direction.RX, channel: str = "0", **kwargs: Any
    ) -> Command:
        """Select an antenna port by name and publish the command."""
        return self._execute_command(self._driver.set_antenna, antenna, "antenna", direction, channel, **kwargs)

    def get_antenna(self, *, direction: Direction = Direction.RX, channel: str = "0", **kwargs: Any) -> Measurement:
        """Query the selected antenna port."""
        return self._execute_measurement(self._driver.get_antenna, "antenna", direction, channel, **kwargs)
