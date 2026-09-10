"""Tests for the generic SDR contract and the RTL-SDR adapter."""

from __future__ import annotations

import copy
import logging
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from instro.lib import Command
from instro.unstable.sdr import Direction, InstroSDR, IQCapture, SDRDriverBase


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

    def set_center_freq(self, frequency_hz: float, **kwargs) -> None:
        self._center_freq_hz = float(frequency_hz)

    def get_center_freq(self, **kwargs) -> float:
        return self._center_freq_hz

    def set_sample_rate(self, sample_rate_hz: float, **kwargs) -> None:
        self._sample_rate_hz = float(sample_rate_hz)

    def get_sample_rate(self, **kwargs) -> float:
        return self._sample_rate_hz

    def set_gain(self, gain_db: float, **kwargs) -> None:
        self._gain_db = float(gain_db)

    def get_gain(self, **kwargs) -> float:
        return self._gain_db

    def set_bandwidth(self, bandwidth_hz: float, **kwargs) -> None:
        self._bandwidth_hz = float(bandwidth_hz)

    def get_bandwidth(self, **kwargs) -> float:
        return self._bandwidth_hz

    def read_iq(self, n_samples: int, **kwargs) -> IQCapture:
        base = np.linspace(0, 1, n_samples, dtype=float)
        samples = base.astype(np.complex128) + 1j * (base * 2.0)
        return IQCapture(
            samples=samples,
            sample_period_ns=1e9 / self._sample_rate_hz,
            center_freq_hz=self._center_freq_hz,
        )


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

    assert measurement.channel_data["rtl.rx0.i"]
    assert measurement.channel_data["rtl.rx0.q"]
    assert len(measurement.timestamps) == 8
    assert measurement.channel_data["rtl.rx0.i"][0] == pytest.approx(0.0)
    assert measurement.channel_data["rtl.rx0.q"][-1] == pytest.approx(2.0)


def test_03_instro_sdr_measure_spectrum_publishes_one_scalar_per_channel() -> None:
    """Regression: the summary emitted 5 values against 1 timestamp, which no publisher can zip."""
    published = []
    publisher = MagicMock()
    publisher.publish.side_effect = lambda data, **kwargs: published.append(data)
    sdr = InstroSDR(name="rtl", driver=_MinimalSDRDriver(), publishers=[publisher])

    measurement = sdr.measure_spectrum(n_samples=1024)

    assert published == [measurement]
    assert set(measurement.channel_data) == {
        "rtl.rx0.spectrum.peak_power_db",
        "rtl.rx0.spectrum.peak_freq_hz",
        "rtl.rx0.spectrum.mean_power_db",
        "rtl.rx0.spectrum.occupied_bw_hz",
    }
    assert len(measurement.timestamps) == 1
    assert all(len(values) == 1 for values in measurement.channel_data.values())


def test_04_instro_sdr_safely_wraps_driver_methods() -> None:
    driver = MagicMock(spec=_MinimalSDRDriver)
    driver.get_sample_rate.return_value = 2_400_000.0
    driver.read_iq.return_value = IQCapture(
        samples=np.array([1 + 2j, 3 + 4j], dtype=np.complex128),
        sample_period_ns=1e9 / 2_400_000.0,
        center_freq_hz=100_000_000.0,
    )
    sdr = InstroSDR(name="rtl", driver=driver)

    measurement = sdr.measure_iq(n_samples=2)

    assert "rtl.rx0.i" in measurement.channel_data
    assert measurement.channel_data["rtl.rx0.i"][0] == pytest.approx(1.0)
    assert measurement.channel_data["rtl.rx0.q"][1] == pytest.approx(4.0)


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

    assert measurement.channel_data == {f"rtl.rx0.{descriptor}": [initial_value]}


def test_06_instro_sdr_getters_publish_to_attached_publishers() -> None:
    published = []
    publisher = MagicMock()
    publisher.publish.side_effect = lambda data, **kwargs: published.append(data)
    driver = _MinimalSDRDriver()
    sdr = InstroSDR(name="rtl", driver=driver, publishers=[publisher])

    sdr.get_gain()

    assert len(published) == 1
    assert published[0].channel_data == {"rtl.rx0.gain": [20.0]}


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
    assert command.channel_data == {f"rtl.rx0.{descriptor}.cmd": value}


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


def test_11_measure_iq_rejects_a_non_positive_sample_period() -> None:
    """A driver that cannot report its timebase must fail loudly rather than fabricate one."""
    driver = MagicMock(spec=_MinimalSDRDriver)
    driver.read_iq.return_value = IQCapture(
        samples=np.array([1 + 2j], dtype=np.complex128), sample_period_ns=0.0, center_freq_hz=1e8
    )
    sdr = InstroSDR(name="rtl", driver=driver)

    with pytest.raises(ValueError, match="non-positive sample period"):
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
    """A sub-nanosecond period collapses samples onto duplicate timestamps; warn, don't fail."""
    driver = MagicMock(spec=_MinimalSDRDriver)
    driver.read_iq.return_value = IQCapture(
        samples=np.zeros(8, dtype=np.complex128), sample_period_ns=0.5, center_freq_hz=1e8
    )
    sdr = InstroSDR(name="rtl", driver=driver)

    with caplog.at_level(logging.WARNING, logger="instro.unstable.sdr.sdr"):
        measurement = sdr.measure_iq(n_samples=8)
        sdr.measure_iq(n_samples=8)

    assert len(measurement.channel_data["rtl.rx0.i"]) == 8
    assert len([r for r in caplog.records if "shorter than the integer nanosecond" in r.getMessage()]) == 1


class _ToneSDRDriver(_MinimalSDRDriver):
    """Emits a single complex tone offset from the centre frequency."""

    def __init__(self, offset_hz: float) -> None:
        super().__init__()
        self._offset_hz = offset_hz

    def read_iq(self, n_samples: int, **kwargs) -> IQCapture:
        n = np.arange(n_samples)
        samples = np.exp(2j * np.pi * self._offset_hz * n / self._sample_rate_hz)
        return IQCapture(
            samples=samples,
            sample_period_ns=1e9 / self._sample_rate_hz,
            center_freq_hz=self._center_freq_hz,
        )


def test_14_measure_spectrum_locates_a_tone_at_its_true_frequency() -> None:
    """Regression: the old summary was |z|^2 resampled to 5 points -- no FFT, no frequency axis."""
    offset_hz = 300_000.0
    sdr = InstroSDR(name="rtl", driver=_ToneSDRDriver(offset_hz))

    measurement = sdr.measure_spectrum(n_samples=1024)

    bin_width_hz = 2_400_000.0 / 1024
    assert measurement.channel_data["rtl.rx0.spectrum.peak_freq_hz"][0] == pytest.approx(
        100_000_000.0 + offset_hz, abs=bin_width_hz
    )
    peak_db = measurement.channel_data["rtl.rx0.spectrum.peak_power_db"][0]
    assert peak_db > measurement.channel_data["rtl.rx0.spectrum.mean_power_db"][0] + 20
    assert measurement.channel_data["rtl.rx0.spectrum.occupied_bw_hz"][0] < 10 * bin_width_hz


def test_15_compute_psd_returns_the_array_without_publishing() -> None:
    """The full spectrum is caller-facing only; it must not reach publishers."""
    published = []
    publisher = MagicMock()
    publisher.publish.side_effect = lambda data, **kwargs: published.append(data)
    sdr = InstroSDR(name="rtl", driver=_ToneSDRDriver(300_000.0), publishers=[publisher])

    freqs, power_db = sdr.compute_psd(n_samples=1024)

    assert published == []
    assert freqs.shape == power_db.shape == (1024,)
    assert np.all(np.diff(freqs) > 0)
    assert freqs[int(np.argmax(power_db))] == pytest.approx(100_300_000.0, abs=2_400_000.0 / 1024)


def test_16_attribute_lookup_before_driver_is_set_does_not_recurse() -> None:
    """Regression: __getattr__ read self._driver unguarded, so a lookup before it was set recursed."""
    half_built = InstroSDR.__new__(InstroSDR)  # _driver not assigned yet

    with pytest.raises(AttributeError):
        half_built.anything  # noqa: B018 -- the attribute access itself is under test

    # copy.copy consults dunders that InstroSDR lacks, which is what tripped the recursion.
    assert copy.copy(InstroSDR(name="rtl", driver=_MinimalSDRDriver())) is not None


def test_17_instro_sdr_does_not_delegate_to_the_driver() -> None:
    """Driver methods stay behind the HAL: delegation bypassed the lock and the publishing path."""
    driver = _MinimalSDRDriver()
    sdr = InstroSDR(name="rtl", driver=driver)

    with pytest.raises(AttributeError, match="InstroSDR"):
        sdr.read_iq  # noqa: B018 -- the attribute access itself is under test

    assert sdr.driver is driver
    assert sdr.driver.read_iq(4).samples.shape == (4,)


def test_18_measure_iq_values_match_per_element_conversion_exactly() -> None:
    """The vectorised conversion must be bit-identical to the per-element form it replaced."""
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(2048) + 1j * rng.standard_normal(2048)).astype(np.complex128)
    driver = MagicMock(spec=_MinimalSDRDriver)
    driver.read_iq.return_value = IQCapture(
        samples=samples, sample_period_ns=1e9 / 2_400_000.0, center_freq_hz=100_000_000.0
    )
    sdr = InstroSDR(name="rtl", driver=driver)

    measurement = sdr.measure_iq(n_samples=2048)

    assert measurement.channel_data["rtl.rx0.i"] == [float(np.real(v)) for v in samples]
    assert measurement.channel_data["rtl.rx0.q"] == [float(np.imag(v)) for v in samples]
    # Publishers require plain Python scalars, not numpy types.
    assert type(measurement.channel_data["rtl.rx0.i"][0]) is float
    assert type(measurement.timestamps[0]) is int


def test_19_a_hardware_timestamp_anchors_the_block() -> None:
    """A driver whose device timed the samples supplies t0; the host clock must not override it."""
    hardware_t0 = 1_700_000_000_000_000_000
    period_ns = 1e9 / 2_400_000.0
    driver = MagicMock(spec=_MinimalSDRDriver)
    driver.read_iq.return_value = IQCapture(
        samples=np.zeros(64, dtype=np.complex128),
        sample_period_ns=period_ns,
        center_freq_hz=100_000_000.0,
        t0_ns=hardware_t0,
    )
    sdr = InstroSDR(name="rtl", driver=driver)

    measurement = sdr.measure_iq(n_samples=64)

    assert measurement.timestamps[0] == hardware_t0
    assert measurement.timestamps[-1] == hardware_t0 + round(63 * period_ns)


def test_20_hardware_timestamps_bypass_the_wall_clock_continuity_guard() -> None:
    """With a device clock, blocks land where the device says, even if that repeats a timeline."""
    period_ns = 1e9 / 2_400_000.0
    driver = MagicMock(spec=_MinimalSDRDriver)
    sdr = InstroSDR(name="rtl", driver=driver)

    driver.read_iq.return_value = IQCapture(
        samples=np.zeros(8, dtype=np.complex128),
        sample_period_ns=period_ns,
        center_freq_hz=1e8,
        t0_ns=5_000,
    )
    first = sdr.measure_iq(n_samples=8)
    second = sdr.measure_iq(n_samples=8)

    assert first.timestamps == second.timestamps == [5_000 + round(i * period_ns) for i in range(8)]


def test_21_measure_iq_takes_the_timebase_from_the_capture_not_a_second_query() -> None:
    """Regression risk: re-querying the rate can disagree with the block that was just read."""
    driver = MagicMock(spec=_MinimalSDRDriver)
    driver.read_iq.return_value = IQCapture(
        samples=np.zeros(16, dtype=np.complex128),
        sample_period_ns=1e9 / 2_400_000.0,
        center_freq_hz=100_000_000.0,
    )
    driver.get_sample_rate.return_value = 999.0  # a stale value the HAL must ignore
    sdr = InstroSDR(name="rtl", driver=driver)

    measurement = sdr.measure_iq(n_samples=16)

    driver.get_sample_rate.assert_not_called()
    assert set(np.diff(measurement.timestamps).tolist()) <= {416, 417}


class _RequiredOnlyDriver(SDRDriverBase):
    """Implements the required tier and nothing else."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    def set_center_freq(self, frequency_hz: float, **kwargs) -> None: ...

    def get_center_freq(self, **kwargs) -> float:
        return 1e8

    def set_sample_rate(self, sample_rate_hz: float, **kwargs) -> None: ...

    def get_sample_rate(self, **kwargs) -> float:
        return 2.4e6

    def read_iq(self, n_samples: int, **kwargs) -> IQCapture:
        return IQCapture(
            samples=np.zeros(n_samples, dtype=np.complex128),
            sample_period_ns=1e9 / 2.4e6,
            center_freq_hz=1e8,
        )


def test_22_a_driver_implementing_only_the_required_tier_is_concrete() -> None:
    """Optional capabilities must not force stubs onto drivers whose hardware lacks them."""
    driver = _RequiredOnlyDriver()

    assert isinstance(driver, SDRDriverBase)
    assert InstroSDR(name="rtl", driver=driver).measure_iq(n_samples=8) is not None


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("set_gain", (10.0,)),
        ("get_gain", ()),
        ("set_gain_mode", (True,)),
        ("get_gain_mode", ()),
        ("get_gain_range", ()),
        ("set_bandwidth", (1e6,)),
        ("get_bandwidth", ()),
        ("set_freq_correction", (10.0,)),
        ("get_freq_correction", ()),
        ("list_antennas", ()),
        ("set_antenna", ("RX2",)),
        ("get_antenna", ()),
        ("get_num_channels", ()),
        ("get_frequency_range", ()),
        ("get_sample_rate_range", ()),
    ],
)
def test_23_unimplemented_optional_methods_say_so(method: str, args: tuple) -> None:
    """An unsupported capability raises NotImplementedError, not AttributeError or silence."""
    driver = _RequiredOnlyDriver()

    with pytest.raises(NotImplementedError):
        getattr(driver, method)(*args)


def test_24_direction_and_channel_reach_the_driver() -> None:
    """The signal path a caller names must be the one the driver is asked about."""
    driver = MagicMock(spec=_MinimalSDRDriver)
    driver.get_center_freq.return_value = 1e8
    sdr = InstroSDR(name="usrp", driver=driver)

    measurement = sdr.get_center_freq(direction=Direction.TX, channel="1")

    driver.get_center_freq.assert_called_once_with(direction=Direction.TX, channel="1")
    assert measurement.channel_data == {"usrp.tx1.center_freq": [1e8]}


def test_25_published_channels_name_the_signal_path() -> None:
    """Multi-path radios need every channel to say which path it came from."""
    sdr = InstroSDR(name="rtl", driver=_MinimalSDRDriver())

    iq = sdr.measure_iq(n_samples=8)
    command = sdr.set_center_freq(1e8)

    assert set(iq.channel_data) == {"rtl.rx0.i", "rtl.rx0.q"}
    assert set(command.channel_data) == {"rtl.rx0.center_freq.cmd"}
