"""Vendor-specific VNA adapters for the unstable VNA interface.

This subpackage contains thin compatibility shims that adapt third-party
instrument drivers to the repository's VNA abstraction. The shims primarily
implement the methods and data model required by `VNADriverBase` while
forwarding device-specific calls to the wrapped vendor driver object.

Submodules
----------
`skrf_rs_zna`
    Rohde & Schwarz ZNA shim built on top of the `skrf` driver stack.
    The concrete class exported here is `RSVNA`, which wraps the underlying
    `skrf.vi.vna.rohde_schwarz.rs_vna.RSVNA` instance and exposes the VNA
    interface expected by higher-level code.

The package re-exports the shim classes so callers can do imports like:

    from instro.unstable.vna.shims import RSVNA
"""

from .skrf_rs_zna import RSVNA

__all__ = ["RSVNA"]
