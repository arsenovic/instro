
import abc
import logging
from instro.lib import InstroError, Instrument
from instro.lib.publishers import Publisher
from instro.lib.types import Command

import numpy as np
try: 
    import skrf 
except ImportError:
    raise InstroError("scikit-rf is required for VNA support. Please install it with `pip install scikit-rf`.")

from collections.abc import Sequence

logger = logging.getLogger(__name__)


class VNADriverBase(abc.ABC):
    """Base class for VNA drivers."""

    @abc.abstractmethod
    def get_sparam(
        self, 
        m:int, 
        n:int, 
        cnum: int|None = None,
         ) -> np.array:
        """
        Get a single S-parameter"""
        ...

    @abc.abstractmethod
    def get_snp_network(
            self,
            ports: Sequence | None = None,
            cnum: int|None = None,
            **kw,
        ) -> skrf.Network:
        """Get the S-parameter network from the VNA in form of a skrf.Network object.
        cnum: The channel number to get the network from. If None, use the active channel or dont use channels if not supported.
        ports: The ports to get the network from. If None, use all ports.
        """
        ...

    @abc.abstractmethod
    def get_frequency(
        self,
        cnum: int|None = None
        ) -> skrf.Frequency:
        """Get the frequency of the VNA."""
        ...
    @abc.abstractmethod
    def set_frequency(
        self,
        freq: skrf.Frequency,
        cnum: int|None = None,

        ) :
        """Set the frequency of the VNA."""
        ...
    

class InstroVNA(Instrument):
    def __init__(
        self,
        name: str,
        driver: VNADriverBase,
        publishers: list[Publisher] | None = None,
        **kwargs,
    ):
        """Initialize an InstroVNA.

        Args:
            name: Channel-name prefix for published data.
            driver: Concrete VNA driver; owns its own transport::

                vna = InstroVNA(
                    "myVNA",
                    driver=Keysight8510C("USB0::0x0957::0x0507::MY44001757::INSTR"),
                )

            publishers: Publishers that receive emitted Measurement/Command data.
            **kwargs: Default tags applied to every emitted Measurement/Command.
                Pass ``dataset_rid="<rid>"`` to auto-create a NominalCorePublisher
                (uses the on-disk 'default' Nominal credential).
        """
        super().__init__(name, publishers=publishers, **kwargs)

        self._driver = driver
        
    #WHY: do we have this as a property ?
    @property
    def driver(self) -> VNADriverBase:
        """The underlying vendor driver """
        return self._driver

    def get_snp_network(
            self,
            cnum: int|None = None,
            ports: Sequence | None = None,
            **kw,
        ) -> skrf.Network:
        """Get the S-parameter network from the VNA in form of a skrf.Network object.
        cnum: The channel number to get the network from. If None, use the active channel or dont use channels if not supported.
        ports: The ports to get the network from. If None, use all ports.
        **kwargs: Additional keyword arguments passed to the underlying Network() constructor.
        """
        return self._driver.get_snp_network(cnum=cnum, ports=ports,**kw)

    @property 
    def s11( self, **kw):
        return self.get_snp_network(ports=[0], **kw)

    @property 
    def s22( self, **kw):
        return self.get_snp_network(ports=[1], **kw)

    def get_twoport(self, **kw):
        return self.get_snp_network(ports=[0, 1], **kw)

    @property
    def get_frequency(self,cnum: int|None = None) -> skrf.Frequency:
        """Get the frequency of the VNA."""
        return self._driver.get_frequency(cnum=cnum)

    @property
    def frequency(self):
        return self._driver.get_frequency()

    @frequency.setter
    def frequency(self,freq):
        return self._driver.set_frequency(freq)

    

    