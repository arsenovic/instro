"""Tests for the RTL-SDR adapter."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import numpy as np
import pytest

from instro.unstable.sdr.drivers import RTLSDR


def _patch_rtlsdr():
    """Patch the vendor class the driver imports, with a device stub carrying sane defaults."""
    patcher = patch("rtlsdr.RtlSdr", autospec=True)
    rtl_sdr_cls = patcher.start()
    device = rtl_sdr_cls.return_value
    device.center_freq = 80_000_000
    device.sample_rate = 1_024_000.0
    device.gain = 0.0
    device.bandwidth = 0
    return patcher, rtl_sdr_cls, device


def test_01_rtlsdr_opens_the_requested_device_index() -> None:
    """Regression: the driver passed index=, which RtlSdr rejects, and silently fell back to device 0."""
    patcher, rtl_sdr_cls, _ = _patch_rtlsdr()
    try:
        RTLSDR(device_index=2).open()
    finally:
        patcher.stop()

    rtl_sdr_cls.assert_called_once_with(device_index=2)


def test_02_rtlsdr_opens_exactly_one_handle() -> None:
    """Regression: a non-zero index constructed a second device and leaked the first."""
    patcher, rtl_sdr_cls, _ = _patch_rtlsdr()
    try:
        RTLSDR(device_index=1).open()
    finally:
        patcher.stop()

    assert rtl_sdr_cls.call_count == 1


def test_03_rtlsdr_propagates_an_open_failure() -> None:
    """Regression: `except TypeError` swallowed the failure and opened a different device instead."""
    patcher = patch("rtlsdr.RtlSdr", autospec=True)
    rtl_sdr_cls = patcher.start()
    rtl_sdr_cls.side_effect = OSError("Could not open SDR (device index = 5)")
    try:
        with pytest.raises(OSError, match="device index = 5"):
            RTLSDR(device_index=5).open()
    finally:
        patcher.stop()


def test_04_rtlsdr_forwards_vendor_kwargs() -> None:
    """serial_number and friends reach RtlSdr without colliding with device_index."""
    patcher, rtl_sdr_cls, _ = _patch_rtlsdr()
    try:
        RTLSDR(device_index=0, serial_number="00000001").open()
    finally:
        patcher.stop()

    rtl_sdr_cls.assert_called_once_with(device_index=0, serial_number="00000001")


def test_05_construction_touches_no_hardware() -> None:
    """Connection params are captured in __init__; the USB handle is taken in open()."""
    patcher, rtl_sdr_cls, _ = _patch_rtlsdr()
    try:
        driver = RTLSDR(device_index=0)
        assert rtl_sdr_cls.call_count == 0

        driver.open()
        assert rtl_sdr_cls.call_count == 1
    finally:
        patcher.stop()


def test_06_rtlsdr_reopens_after_close() -> None:
    """Regression: close() left a dead handle in place and open() only logged, so reuse hit a closed device."""
    patcher, rtl_sdr_cls, device = _patch_rtlsdr()
    try:
        driver = RTLSDR(device_index=0)
        driver.open()
        driver.close()
        device.close.assert_called_once()

        driver.open()
        assert rtl_sdr_cls.call_count == 2
    finally:
        patcher.stop()


def test_07_rtlsdr_open_is_idempotent() -> None:
    """A second open() on a live driver must not take a second handle."""
    patcher, rtl_sdr_cls, _ = _patch_rtlsdr()
    try:
        driver = RTLSDR(device_index=0)
        driver.open()
        driver.open()
    finally:
        patcher.stop()

    assert rtl_sdr_cls.call_count == 1


def test_08_rtlsdr_raises_a_clear_error_when_not_open() -> None:
    """Using a closed driver must say so, not crash inside the vendor library."""
    driver = RTLSDR(device_index=0)

    with pytest.raises(RuntimeError, match="not open"):
        driver.read_iq(16)
    with pytest.raises(RuntimeError, match="not open"):
        driver.get_center_freq()


def test_09_driver_imports_without_the_vendor_sdk_installed() -> None:
    """The vendor SDK belongs to instro-unstable, not core instro: importing must not require it."""
    module = importlib.import_module("instro.unstable.sdr.drivers.rtl_sdr")

    with patch.dict(sys.modules, {"rtlsdr": None}):
        importlib.reload(module)
        driver = module.RTLSDR(device_index=0)  # construction stays SDK-free

        with pytest.raises(ImportError):
            driver.open()  # only the USB handle needs the SDK

    importlib.reload(module)


@pytest.mark.hardware
def test_10_rtlsdr_reads_iq_from_a_connected_dongle() -> None:
    """Requires one RTL-SDR on USB. Verifies open, configure, a real IQ read, and reopen."""
    sdr = RTLSDR(device_index=0)
    try:
        sdr.open()
        sdr.set_center_freq(89_700_000.0)
        sdr.set_sample_rate(2_400_000.0)

        assert sdr.get_center_freq() == pytest.approx(89_700_000.0, rel=1e-4)
        assert sdr.get_sample_rate() == pytest.approx(2_400_000.0, rel=1e-4)

        samples = sdr.read_iq(4096)
        assert samples.shape == (4096,)
        assert np.iscomplexobj(samples)
        assert np.any(samples != 0)

        sdr.close()
        sdr.open()
        assert sdr.read_iq(1024).shape == (1024,)
    finally:
        sdr.close()
