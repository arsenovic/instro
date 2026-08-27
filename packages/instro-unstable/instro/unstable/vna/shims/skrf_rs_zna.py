from __future__ import annotations

from skrf.vi.vna.rohde_schwarz.rs_vna import RSVNA as SKRF_RSVNA

from ..vna import VNADriverBase


class RSVNA(VNADriverBase):
    def __init__(self, *args, **kwargs):
        self._shim_driver = SKRF_RSVNA(*args, **kwargs)

    @property
    def shim_driver(self):
        """Expose the wrapped skrf driver for inspection and debugging."""
        return self._shim_driver

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying shim_driver."""
        try:
            return getattr(self._shim_driver, name)
        except AttributeError as exc:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}") from exc

    def __dir__(self):
        """Include wrapped-driver attributes in tab-completion and dir()."""
        return sorted(set(super().__dir__()) | set(dir(self._shim_driver)))

    def open(self):
        # TODO: this shoudl access somting in shim_driver
        return None

    def close(self):
        # TODO: this shoudl access somting in shim_driver
        return None

    def get_ch(self, ch):
        """Return the underlying skrf channel instance for ``ch``."""
        return getattr(self._shim_driver, f"ch{ch}")

    def _call_channel_attr(self, ch, attr_name, *args, **kwargs):
        channel = self.get_ch(ch)
        value = getattr(channel, attr_name)
        if callable(value):
            return value(*args, **kwargs)
        return value

    def _set_channel_attr(self, ch, attr_name, value, *args, **kwargs):
        channel = self.get_ch(ch)
        target = getattr(channel, attr_name)
        if callable(target):
            return target(value, *args, **kwargs)
        setattr(channel, attr_name, value)
        return value

    def get_freq_start(self, ch=None, *args, **kwargs):
        return self._call_channel_attr(ch, "freq_start", *args, **kwargs)

    def set_freq_start(self, ch=None, freq=None, *args, **kwargs):
        return self._set_channel_attr(ch, "freq_start", freq, *args, **kwargs)

    def get_freq_stop(self, ch=None, *args, **kwargs):
        return self._call_channel_attr(ch, "freq_stop", *args, **kwargs)

    def set_freq_stop(self, ch=None, freq=None, *args, **kwargs):
        return self._set_channel_attr(ch, "freq_stop", freq, *args, **kwargs)

    def get_freq_npoints(self, ch=None, *args, **kwargs):
        return self._call_channel_attr(ch, "npoints", *args, **kwargs)

    def set_freq_npoints(self, ch=None, npoints=None, *args, **kwargs):
        return self._set_channel_attr(ch, "npoints", npoints, *args, **kwargs)

    def get_nports(self, ch=None, *args, **kwargs):
        return self._call_channel_attr(ch, "nports", *args, **kwargs)

    def get_smat(self, m, n, ch=None, *args, **kwargs):
        channel = self.get_ch(ch)
        return channel.s_data(m, n, *args, **kwargs)
