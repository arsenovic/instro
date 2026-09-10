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
from typing import Any, Callable

import numpy as np

from instro.lib import Command, Instrument, Measurement
from instro.lib.instrument import publish_command, publish_measurement

logger = logging.getLogger(__name__)


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
    def read_iq(self, n_samples: int) -> np.ndarray:
        """Read a block of complex IQ samples."""
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

    def _iq_timestamps(self, t_read_ns: int, sample_rate_hz: float, length: int) -> list[int]:
        """Nanosecond timestamps for one IQ block, spaced at the device's actual sample period."""
        if sample_rate_hz <= 0:
            raise ValueError(f"driver reported a non-positive sample rate: {sample_rate_hz}")
        if sample_rate_hz > 1e9 and not self._sample_period_warning_issued:
            self._sample_period_warning_issued = True
            logger.warning(
                "Sample rate %s Sa/s is faster than the 1 GSa/s that integer-nanosecond timestamps "
                "can resolve; samples in each block will share timestamps.",
                sample_rate_hz,
            )

        period_ns = 1e9 / sample_rate_hz
        offsets = [round(i * period_ns) for i in range(length)]

        t0 = t_read_ns - offsets[-1]
        if self._last_iq_timestamp is not None and t0 <= self._last_iq_timestamp:
            t0 = self._last_iq_timestamp + round(period_ns)

        timestamps = [t0 + offset for offset in offsets]
        self._last_iq_timestamp = timestamps[-1]
        return timestamps

    @publish_measurement
    def measure_iq(self, n_samples: int = 1024, **kwargs: Any) -> Measurement | None:
        """Return a ``Measurement`` containing one IQ buffer block.

        The measurement contains real and imaginary channels as arrays with a common
        timestamp vector. This keeps the payload structured and efficient while
        avoiding one published object per individual sample.
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        with self._resource_lock:
            data = np.asarray(self._driver.read_iq(n_samples), dtype=np.complex128)
            sample_rate_hz = float(self._driver.get_sample_rate())
            timestamp = time.time_ns()

            if data.size == 0:
                return None

            timestamps = self._iq_timestamps(timestamp, sample_rate_hz, len(data))

        complex_values = data
        return Measurement(
            channel_data={
                f"{self.name}.i": [float(np.real(v)) for v in complex_values],
                f"{self.name}.q": [float(np.imag(v)) for v in complex_values],
            },
            timestamps=timestamps,
            tags={**self.default_tags, **kwargs},
        )

    @publish_measurement
    def measure_spectrum(self, n_samples: int = 1024, **kwargs: Any) -> Measurement | None:
        """Return a compact power-spectrum summary for the selected IQ window."""
        iq = self.measure_iq(n_samples=n_samples, **kwargs)
        if iq is None:
            return None

        i_vals = np.asarray(iq.channel_data[f"{self.name}.i"], dtype=float)
        q_vals = np.asarray(iq.channel_data[f"{self.name}.q"], dtype=float)
        power = np.abs(i_vals + 1j * q_vals) ** 2

        n_bins = min(5, len(power))
        if n_bins == 0:
            return None

        x = np.linspace(0, len(power) - 1, n_bins, dtype=float)
        y = np.interp(x, np.arange(len(power)), power)
        timestamp = time.time_ns()

        return Measurement(
            channel_data={f"{self.name}.spectrum": [float(p) for p in y]},
            timestamps=[timestamp],
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

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the underlying driver."""
        attr = getattr(self._driver, name)
        return attr
