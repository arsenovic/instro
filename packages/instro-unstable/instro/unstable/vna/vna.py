import abc
import logging
import time
import threading
from typing import Any, Callable, get_type_hints
from collections.abc import Sequence
from numbers import Number

import numpy as np
import skrf
from functools import wraps

from instro.lib import InstroError, Instrument
from instro.lib.publishers import Publisher
from instro.lib.types import Command, Measurement
from instro.lib.instrument import publish_command, publish_measurement
from .types import SweepType, NetworkFileFormat
from .storage import DiskStorage, Storage
from .external import network_to_dict

logger = logging.getLogger(__name__)


def hint_returns_numeric(func):
    """Return whether a callable is annotated to return a numeric type.

    This is used to detect measurement getters that expose scalar numeric
    values, such as ``float`` or ``int`` returns.
    """
    t = get_type_hints(func).get("return", None)
    return isinstance(t, type) and issubclass(t, Number)


class VNADriverBase(abc.ABC):
    """Base class for VNA drivers.

    Required abstract methods:
        - ``get_freq_start(ch: int | None = None) -> float``
        - ``get_freq_stop(ch: int | None = None) -> float``
        - ``get_freq_npoints(ch: int | None = None) -> int``
        - ``get_nports(ch: int | None = None) -> int``
        - ``get_smat(m: int, n: int, ch: int | None = None) -> np.ndarray``

    Notes
    -----
    This is a flat representation with regard to channels, meaning it does
    not use nested channel objects; instead, ``ch`` is a simple argument to
    all relevant methods.
    """

    # TODO: have a clever way to pass `ch` everwhere without seeing it all the time
    # maybe a `channalize` decorator
    @abc.abstractmethod
    def get_freq_start(self, ch: int | None = None) -> float:
        """Get the start frequency of the VNA in Hz."""
        ...

    def set_freq_start(self, ch: int | None = None, freq: float | None = None) -> float:
        """Set the start frequency of the VNA in Hz."""
        ...

    @abc.abstractmethod
    def get_freq_stop(self, ch: int | None = None) -> float:
        """Get the stop frequency of the VNA in Hz."""
        ...

    def set_freq_stop(self, ch: int | None = None, freq: float | None = None) -> float:
        """Set the stop frequency of the VNA in Hz."""
        ...

    def get_freq_span(self, ch: int | None = None) -> float:
        """Get the frequency span of the VNA in Hz."""
        ...

    def set_freq_span(self, ch: int | None = None, freq: float | None = None) -> float:
        """Set the frequency span of the VNA in Hz."""
        ...

    def get_freq_center(self, ch: int | None = None) -> float:
        """Get the center frequency of the VNA in Hz."""
        ...

    def set_freq_center(self, ch: int | None = None, freq: float | None = None) -> float:
        """Set the center frequency of the VNA in Hz."""
        ...

    @abc.abstractmethod
    def get_freq_npoints(self, ch: int | None = None) -> int:
        """Get the number of frequency points of the VNA sweep."""
        ...

    def set_freq_npoints(self, ch: int | None = None, npoints: int | None = None) -> int:
        """Set the number of frequency points of the VNA sweep."""
        ...

    @abc.abstractmethod
    def get_nports(self, ch: int | None = None) -> int:
        """Get the number of ports of the VNA."""
        ...

    @abc.abstractmethod
    def get_smat(
        self,
        m: int,
        n: int,
        ch: int | None = None,
    ) -> np.array:
        """
        Get a single S-parameter as a complex numpy array.
        m: The row index of the S-parameter (0-based).
        n: The column index of the S-parameter (0-based).
        ch: The channel number to get the S-parameter from. If None, use the active channel or dont use channels if not supported.
        """
        ...

    def get_frequency(
        self,
        ch: int | None = None,
        unit: str = "ghz",
        sweep_type: SweepType = "LIN",
    ) -> skrf.Frequency:
        """Get the frequency of the VNA."""
        if sweep_type == "LIN":
            frequency = skrf.Frequency(
                start=self.get_freq_start(ch=ch),
                stop=self.get_freq_stop(ch=ch),
                npoints=self.get_freq_npoints(ch=ch),
            )
            frequency.unit = unit
        else:
            raise NotImplementedError
        return frequency

    def set_frequency(
        self,
        freq: skrf.Frequency,
        ch: int | None = None,
    ):
        """Set the frequency of the VNA."""
        self.set_freq_start(ch=ch, freq=freq.start)
        self.set_freq_stop(ch=ch, freq=freq.stop)
        self.set_freq_npoints(ch=ch, npoints=freq.npoints)

    @property
    def frequency(self):
        return self.get_frequency()

    @frequency.setter
    def frequency(self, freq):
        return self.set_frequency(freq)

    def get_network(
        self,
        ports: Sequence | None = None,
        ch: int | None = None,
        **kw,
    ) -> skrf.Network:
        """Get a network from the VNA as an ``skrf.Network`` object.

        Args:
            ports: Port indices to include in the network. If ``None``, all
                ports reported by the instrument are used.
            ch: Channel number to query. If ``None``, the active channel is
                used when supported.
            **kw: Additional keyword arguments passed to the underlying
                ``skrf.Network`` constructor.

        Returns:
            A network object containing the measured S-parameters.
        """
        frequency = self.get_frequency(ch=ch)
        if ports is None:
            ports = range(self.get_nports())

        # iterate over ports and populate the s-parameter matrix
        s = np.zeros((len(frequency.f), len(ports), len(ports)), dtype=complex)
        for i, m in enumerate(ports):
            for j, n in enumerate(ports):
                s[:, i, j] = self.get_smat(m, n, ch=ch)
        network = skrf.Network(frequency=frequency, s=s, **kw)
        return network

    def get_s(self, m: int, n: int, ch: int | None = None, **kw) -> skrf.Network:
        """Get a single S-parameter as a one-port network object.

        Args:
            m: Row index of the S-parameter (0-based).
            n: Column index of the S-parameter (0-based).
            ch: Channel number to query. If ``None``, the active channel is
                used when supported.
            **kw: Extra keyword arguments forwarded to ``skrf.Network``.

        Returns:
            A one-port network containing the selected S-parameter.
        """
        frequency = self.get_frequency(ch=ch)
        s = self.get_smat(m, n, ch=ch)
        network = skrf.Network(frequency=frequency, s=s[:, np.newaxis, np.newaxis], **kw)
        return network

    @property
    def s11(self):
        return self.get_s(m=0, n=0)

    @property
    def s22(self):
        return self.get_s(m=1, n=1)

    @property
    def s21(self):
        return self.get_s(m=1, n=0)

    @property
    def s12(self):
        return self.get_s(m=0, n=1)


class InstroVNA(Instrument):
    def __init__(
        self,
        name: str,
        driver: VNADriverBase,
        publishers: list[Publisher] | None = None,
        storage: Storage = DiskStorage(),
        **kwargs,
    ):
        """
        High-level VNA wrapper around a vendor driver.

        This class exposes the underlying driver through the Instro instrument
        interface and wraps numeric getter calls as published measurements.

        Args:
            name: Human-readable name for the instrument.
            driver: Vendor-specific VNA driver implementing the
                ``VNADriverBase`` interface.
            publishers: Optional publishers for measurements and commands.
            storage: Storage backend used for saving network data.
            **kwargs: Extra keyword arguments passed to the base
                ``Instrument`` initializer.

        Examples
        --------
        >>> from instro.unstable.vna.vna import InstroVNA
        >>> from instro.unstable.vna.drivers.nanovna_v2clone import NanoVNAv2Clone
        >>> from instro.unstable.vna.storage import DiskStorage
        >>>
        >>> vna = InstroVNA(
        ...     name='bob',
        ...     driver=NanoVNAv2Clone(port='/dev/ttyACM0'),
        ...     storage=DiskStorage(),  # default path is a tempdir
        ... )
        >>> network = vna.get_network()
        >>> vna.save_network('billy')
        >>> vna.measure_network()

        Notes
        -----
        Any method on ``driver`` whose name starts with ``get_`` and whose return
        annotation is numeric is wrapped and published as an Instro
        ``Measurement``.
        """
        super().__init__(name, publishers=publishers, **kwargs)
        self._driver = driver
        self._resource_lock = threading.Lock()
        self._storage = storage

    # this is general and should be inherited
    @publish_measurement
    def _execute_measurement(
        self,
        driver_method: Callable,
        driver_kwargs: dict[str, Any] | None = None,
        channel: int = 1,
        *args,
        **kwargs,
    ) -> Measurement | None:
        """Execute a driver measurement method and return a Measurement for the read value."""
        name = driver_method.__name__
        channel = name.split("_")[1]  # get_freq_start  # freq is the channel
        # could do something with kwargs to assembly channel name better
        with self._resource_lock:
            data = driver_method(**(driver_kwargs or {}))
            timestamp = time.time_ns()

        channel = f"ch{driver_method.__name__}"

        return self._package_measurement(channel=channel, data=data, timestamp=timestamp, **kwargs)

    @publish_command
    def _execute_command(
        self,
        driver_method: Callable,
        value: Any,
        channel: int = 1,
        **kwargs,
    ) -> Command:
        """Execute a driver command method and return a Command for the published value."""
        with self._resource_lock:
            driver_method(value, channel=channel)
            timestamp = time.time_ns()

        channel = f"ch{driver_method.__name__}.cmd"
        return self._package_command(channel=channel, data=value, timestamp=timestamp, **kwargs)

    @property
    def driver(self) -> VNADriverBase:
        """The underlying vendor driver"""
        return self._driver

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying driver.

        If the attribute on the driver is callable, return a wrapper that
        acquires the InstroVNA resource lock before calling the driver
        method. Non-callable attributes are returned directly.
        """
        driver = self._driver
        if hasattr(driver, name):
            attr = getattr(driver, name)
            if callable(attr):
                if name.startswith("get_") and hint_returns_numeric(attr):

                    @wraps(attr)
                    def _wrapped(**kwargs):
                        return self._execute_measurement(driver_method=attr, driver_kwargs=kwargs)
                elif name.startswith("set_"):  # TODO: and set args_are_numeric(attr):
                    _wrapped = attr  # TODO: wrap set_ methods to execute   commands
                else:
                    _wrapped = attr
                return _wrapped
            return attr
        raise AttributeError(f"{type(self).__name__} object has no attribute {name}")

    def save_network(
        self,
        name: str | None = None,
        ports: Sequence[int] | None = None,
        ch: int | None = None,
        format: NetworkFileFormat = "SNP",
        **kw,
    ) -> skrf.Network:
        """measure a network, save it to self._storage, and return a Measurement
        with channel_data=path to the saved file."""

        with self._resource_lock:
            timestamp = time.time_ns()
            network = self._driver.get_network(ports=ports, ch=ch, **kw)
        if name is None:
            name = f"{self.name}_network_{timestamp}"
        path = self._storage.get_path_for_filename(f"{name}.s{network.nports}p")
        if format == "SNP":
            network.write_touchstone(path)
        else:
            raise NotImplementedError

        return Measurement(
            channel_data={f"{self.name}.save_network": str(path)},
            timestamps=[timestamp],
            tags={**self.default_tags},
        )

    def measure_network(self, name: str | None = None, port: int = 0, ch: int | None = None, **kw) -> skrf.Network:
        """get and save a network to self._storage and return a Measurement with the path to the saved file."""
        # TODO: make port allow multiple 'ports', requires network_to_dict() to support this first
        with self._resource_lock:
            timestamp = time.time_ns()
            network = self._driver.get_network(ports=[port], ch=ch, **kw)
        if name is None:
            name = f"{self.name}_network_{timestamp}"

        data = network_to_dict(network)

        return Measurement(
            channel_data={f"{self.name}.save_network": data},
            timestamps=[timestamp],
            tags={**self.default_tags},
        )
