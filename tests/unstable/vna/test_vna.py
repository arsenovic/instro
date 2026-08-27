"""Higher-level tests for the VNA wrapper using a simple in-memory driver."""

from __future__ import annotations

from pathlib import Path

import pytest

import instro.unstable.vna.shims.skrf_rs_zna as skrf_rs_zna_module
from instro.unstable.vna.shims.skrf_rs_zna import RSVNA
from instro.unstable.vna.storage import DiskStorage
from instro.unstable.vna.vna import InstroVNA
from tests.unstable.vna.simulated import DEFAULT_NPOINTS, DEFAULT_NPORTS, SimulatedSKRFRSVNA, SimulatedVNA


@pytest.fixture
def vna() -> InstroVNA:
    driver = SimulatedVNA(npoints=DEFAULT_NPOINTS, nports=DEFAULT_NPORTS)
    return InstroVNA(name="ut", driver=driver, storage=DiskStorage())


@pytest.fixture
def shim_vna(monkeypatch: pytest.MonkeyPatch) -> RSVNA:
    monkeypatch.setattr(skrf_rs_zna_module, "SKRF_RSVNA", SimulatedSKRFRSVNA)
    return RSVNA(start_hz=1_000_000_000.0, stop_hz=2_000_000_000.0, npoints=DEFAULT_NPOINTS, nports=DEFAULT_NPORTS)


def test_instro_vna_get_freq_start_wraps_numeric_getter(vna: InstroVNA) -> None:
    measurement = vna.get_freq_start(ch=1)
    assert "ut.chget_freq_start" in measurement.channel_data
    assert measurement.channel_data["ut.chget_freq_start"] == pytest.approx([1_000_000_000.0])


def test_instro_vna_get_network_and_get_s_build_expected_shapes(vna: InstroVNA) -> None:
    network = vna.get_network(ports=list(range(DEFAULT_NPORTS)), ch=1)
    assert network.s.shape == (DEFAULT_NPOINTS, DEFAULT_NPORTS, DEFAULT_NPORTS)
    assert network.nports == DEFAULT_NPORTS

    s11 = vna.get_s(0, 0, ch=1)
    assert s11.s.shape == (DEFAULT_NPOINTS, 1, 1)


def test_instro_vna_save_network_writes_touchstone_file(vna: InstroVNA, tmp_path: Path) -> None:
    vna._storage = DiskStorage(path=str(tmp_path))

    measurement = vna.save_network(name="test_network", ports=[0, 1], ch=1)

    saved_path = Path(measurement.channel_data["ut.save_network"])
    assert saved_path.name.endswith(".s2p")
    assert saved_path.exists()


def test_instro_vna_measure_network_returns_serialized_data(vna: InstroVNA) -> None:
    measurement = vna.measure_network(name="payload", port=0, ch=1)

    payload = measurement.channel_data["ut.save_network"]
    assert set(payload) >= {"s_db", "f_hz", "f_unit"}
    assert payload["f_hz"] == pytest.approx(vna.driver.get_frequency().f.tolist())


def test_skrf_rs_zna_shim_implements_vna_contract(shim_vna: RSVNA) -> None:
    assert shim_vna.shim_driver.ch1.npoints == DEFAULT_NPOINTS
    assert shim_vna.get_freq_start(ch=1) == pytest.approx(1_000_000_000.0)
    assert shim_vna.get_freq_stop(ch=1) == pytest.approx(2_000_000_000.0)
    assert shim_vna.get_freq_npoints(ch=1) == DEFAULT_NPOINTS
    assert shim_vna.get_nports(ch=1) == DEFAULT_NPORTS

    network = shim_vna.get_network(ports=[0, 1], ch=1)
    assert network.s.shape == (DEFAULT_NPOINTS, DEFAULT_NPORTS, DEFAULT_NPORTS)
    assert "ch1" in dir(shim_vna)
    assert hasattr(shim_vna, "shim_driver")
