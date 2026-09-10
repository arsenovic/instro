"""Tests for the generic SDR contract and the RTL-SDR adapter."""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from instro.lib import Command
from instro.unstable.sdr import InstroSDR, SDRDriverBase


class _MinimalSDRDriver(SDRDriverBase):
    def __init__(self) -> None:
        self._center_freq_hz = 100_000_000.0
        self._sample_rate_hz = 2_400_000.0
        self._gain_db = 20.0
        self._bandwidth_hz = 1_000_000.0

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def set_center_freq(self, frequency_hz: float) -> None:
        self._center_freq_hz = float(frequency_hz)

    def get_center_freq(self) -> float:
        return self._center_freq_hz

    def set_sample_rate(self, sample_rate_hz: float) -> None:
        self._sample_rate_hz = float(sample_rate_hz)

    def get_sample_rate(self) -> float:
        return self._sample_rate_hz

    def set_gain(self, gain_db: float) -> None:
        self._gain_db = float(gain_db)

    def get_gain(self) -> float:
        return self._gain_db

    def set_bandwidth(self, bandwidth_hz: float) -> None:
        self._bandwidth_hz = float(bandwidth_hz)

    def get_bandwidth(self) -> float:
        return self._bandwidth_hz

    def read_iq(self, n_samples: int) -> np.ndarray:
        base = np.linspace(0, 1, n_samples, dtype=float)
        return base.astype(np.complex128) + 1j * (base * 2.0)


def test_01_sdr_driver_base_requires_implementation() -> None:
    with pytest.raises(TypeError):
        SDRDriverBase()  # type: ignore[abstract]

    class _Incomplete(SDRDriverBase):
        def open(self) -> None:
            pass

        def close(self) -> None:
            pass

    with pytest.raises(TypeError):
        _Incomplete()  # type: ignore[abstract]

    assert isinstance(_MinimalSDRDriver(), SDRDriverBase)


def test_02_instro_sdr_measure_iq_packages_a_buffer_as_measurement() -> None:
    driver = _MinimalSDRDriver()
    sdr = InstroSDR(name="rtl", driver=driver)

    measurement = sdr.measure_iq(n_samples=8)

    assert measurement.channel_data["rtl.i"]
    assert measurement.channel_data["rtl.q"]
    assert len(measurement.timestamps) == 8
    assert measurement.channel_data["rtl.i"][0] == pytest.approx(0.0)
    assert measurement.channel_data["rtl.q"][-1] == pytest.approx(2.0)


def test_03_instro_sdr_measure_spectrum_returns_power_summary() -> None:
    driver = _MinimalSDRDriver()
    sdr = InstroSDR(name="rtl", driver=driver)

    measurement = sdr.measure_spectrum(n_samples=8)

    assert "rtl.spectrum" in measurement.channel_data
    assert len(measurement.channel_data["rtl.spectrum"]) == 5
    assert all(value >= 0.0 for value in measurement.channel_data["rtl.spectrum"])


def test_04_instro_sdr_safely_wraps_driver_methods() -> None:
    driver = MagicMock(spec=_MinimalSDRDriver)
    driver.get_sample_rate.return_value = 2_400_000.0
    driver.read_iq.return_value = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
    sdr = InstroSDR(name="rtl", driver=driver)

    measurement = sdr.measure_iq(n_samples=2)

    assert "rtl.i" in measurement.channel_data
    assert measurement.channel_data["rtl.i"][0] == pytest.approx(1.0)
    assert measurement.channel_data["rtl.q"][1] == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("getter_name", "descriptor", "initial_value"),
    [
        ("get_center_freq", "center_freq", 100_000_000.0),
        ("get_sample_rate", "sample_rate", 2_400_000.0),
        ("get_gain", "gain", 20.0),
        ("get_bandwidth", "bandwidth", 1_000_000.0),
    ],
)
def test_05_instro_sdr_getters_publish_as_measurement(getter_name: str, descriptor: str, initial_value: float) -> None:
    """A read publishes as Measurement, not Command -- same categorical convention as every other category."""
    driver = _MinimalSDRDriver()
    sdr = InstroSDR(name="rtl", driver=driver)

    measurement = getattr(sdr, getter_name)()

    assert measurement.channel_data == {f"rtl.{descriptor}": [initial_value]}


def test_06_instro_sdr_getters_publish_to_attached_publishers() -> None:
    published = []
    publisher = MagicMock()
    publisher.publish.side_effect = lambda data, **kwargs: published.append(data)
    driver = _MinimalSDRDriver()
    sdr = InstroSDR(name="rtl", driver=driver, publishers=[publisher])

    sdr.get_gain()

    assert len(published) == 1
    assert published[0].channel_data == {"rtl.gain": [20.0]}


@pytest.mark.parametrize(
    ("setter_name", "value", "descriptor"),
    [
        ("set_center_freq", 89_700_000.0, "center_freq"),
        ("set_sample_rate", 2_400_000.0, "sample_rate"),
        ("set_gain", 30.0, "gain"),
        ("set_bandwidth", 1_500_000.0, "bandwidth"),
    ],
)
def test_07_instro_sdr_setters_publish_a_command(setter_name: str, value: float, descriptor: str) -> None:
    """Regression: setters packaged a Command but never published it -- every set_* was dropped."""
    published = []
    publisher = MagicMock()
    publisher.publish.side_effect = lambda data, **kwargs: published.append(data)
    sdr = InstroSDR(name="rtl", driver=_MinimalSDRDriver(), publishers=[publisher])

    command = getattr(sdr, setter_name)(value)

    assert isinstance(command, Command)
    assert published == [command]
    assert command.channel_data == {f"rtl.{descriptor}.cmd": value}


def test_08_measure_iq_spaces_timestamps_at_the_device_sample_period() -> None:
    """Regression: spacing was hardcoded to 1 ms (1 kSa/s) regardless of the real sample rate."""
    driver = _MinimalSDRDriver()
    driver.set_sample_rate(2_400_000.0)
    sdr = InstroSDR(name="rtl", driver=driver)

    measurement = sdr.measure_iq(n_samples=64)

    # 416.667 ns does not fit in whole ns, so spacing dithers between the two neighbours.
    period_ns = 1e9 / 2_400_000.0
    spacings = {b - a for a, b in zip(measurement.timestamps, measurement.timestamps[1:])}
    assert spacings <= {416, 417}
    assert 1_000_000 not in spacings
    assert measurement.timestamps[-1] - measurement.timestamps[0] == pytest.approx(63 * period_ns, abs=1)

    # The read returns after the samples were taken, so the block ends at "now", not starts there.
    assert measurement.timestamps[-1] <= time.time_ns()


def test_09_measure_iq_blocks_never_overlap_in_time() -> None:
    """Regression: every block re-anchored to the wall clock, so rapid blocks overlapped."""
    driver = _MinimalSDRDriver()
    driver.set_sample_rate(2_400_000.0)
    sdr = InstroSDR(name="rtl", driver=driver)

    first = sdr.measure_iq(n_samples=4096)
    second = sdr.measure_iq(n_samples=4096)

    assert second.timestamps[0] > first.timestamps[-1]


def test_10_measure_iq_re_anchors_after_a_real_gap() -> None:
    """A dropout longer than one block stays visible as a gap instead of being papered over."""
    driver = _MinimalSDRDriver()
    driver.set_sample_rate(2_400_000.0)
    sdr = InstroSDR(name="rtl", driver=driver)

    first = sdr.measure_iq(n_samples=8)
    dt = round(1e9 / 2_400_000.0)
    time.sleep(0.01)
    second = sdr.measure_iq(n_samples=8)

    assert second.timestamps[0] - first.timestamps[-1] > dt


def test_11_measure_iq_rejects_a_non_positive_sample_rate() -> None:
    """A driver that cannot report its rate must fail loudly rather than fabricate a timebase."""
    driver = MagicMock(spec=_MinimalSDRDriver)
    driver.get_sample_rate.return_value = 0.0
    driver.read_iq.return_value = np.array([1 + 2j], dtype=np.complex128)
    sdr = InstroSDR(name="rtl", driver=driver)

    with pytest.raises(ValueError, match="non-positive sample rate"):
        sdr.measure_iq(n_samples=1)


def test_12_measure_iq_timestamps_track_the_exact_sample_period() -> None:
    """Regression: a period rounded to whole ns drifts 800 ppm at RTL-SDR rates."""
    rate, n_samples = 2_400_000.0, 100_000
    driver = _MinimalSDRDriver()
    driver.set_sample_rate(rate)
    sdr = InstroSDR(name="rtl", driver=driver)

    measurement = sdr.measure_iq(n_samples=n_samples)

    period_ns = 1e9 / rate
    t0 = measurement.timestamps[0]
    worst_ns = max(abs((t - t0) - i * period_ns) for i, t in enumerate(measurement.timestamps))
    assert worst_ns <= 1.0  # rounding a 417 ns period instead would drift ~33 us over this block


def test_13_measure_iq_warns_once_on_a_sub_nanosecond_sample_period(caplog) -> None:
    """Above 1 GSa/s integer-ns timestamps collapse samples onto duplicates; warn, don't fail."""
    driver = MagicMock(spec=_MinimalSDRDriver)
    driver.get_sample_rate.return_value = 2e9
    driver.read_iq.return_value = np.zeros(8, dtype=np.complex128)
    sdr = InstroSDR(name="rtl", driver=driver)

    with caplog.at_level(logging.WARNING, logger="instro.unstable.sdr.sdr"):
        measurement = sdr.measure_iq(n_samples=8)
        sdr.measure_iq(n_samples=8)

    assert len(measurement.channel_data["rtl.i"]) == 8
    assert len([r for r in caplog.records if "1 GSa/s" in r.getMessage()]) == 1
