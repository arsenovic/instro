"""Example: InstroSDR w/ RTL-SDR.

Tunes a dongle to an FM broadcast station, captures one IQ block, and publishes
both the raw IQ and the spectrum summary to a dataset (Nominal Core publisher).
"""

import numpy as np

from instro.lib.publishers import NominalCorePublisher
from instro.unstable.sdr import InstroSDR
from instro.unstable.sdr.drivers import RTLSDR

NOMINAL_DATASET_RID = ""  # REPLACE WITH YOUR OWN
CENTER_FREQ_HZ = 89.7e6  # REPLACE WITH A STATION IN YOUR AREA
SAMPLE_RATE_HZ = 2.4e6
GAIN_DB = 30.0
N_SAMPLES = 262144  # must be a multiple of RTLSDR.READ_GRANULARITY (256)

sdr = InstroSDR(
    name="rtl",
    driver=RTLSDR(device_index=0),
    publishers=[NominalCorePublisher(NOMINAL_DATASET_RID)],
)

# The driver captures connection settings on construction; the USB handle is taken in open().
sdr.open()
try:
    # Each setter publishes a `.cmd` channel recording what was requested.
    sdr.set_center_freq(CENTER_FREQ_HZ)
    sdr.set_sample_rate(SAMPLE_RATE_HZ)
    sdr.set_gain(GAIN_DB)

    # Getters publish a Measurement with the value the device actually accepted, which
    # can differ from the request: tuners quantize both frequency and sample rate.
    print(f"tuned to {sdr.get_center_freq().latest / 1e6:.4f} MHz")
    print(f"sampling at {sdr.get_sample_rate().latest:,.0f} Sa/s")

    # One IQ block, published as paired .i and .q channels sharing a timebase derived
    # from the device's own sample rate.
    iq = sdr.measure_iq(n_samples=N_SAMPLES)
    duration_s = (iq.timestamps[-1] - iq.timestamps[0]) / 1e9
    print(f"captured {len(iq.timestamps):,} samples spanning {duration_s * 1e3:.3f} ms")

    # Reassemble the complex signal: I is the real part, Q the imaginary part.
    # Swapping them conjugates the signal and mirrors the spectrum about the center.
    z = np.asarray(iq.channel_data["rtl.i"]) + 1j * np.asarray(iq.channel_data["rtl.q"])
    print(f"mean power {10 * np.log10(np.mean(np.abs(z) ** 2)):.1f} dB")

    # Scalar spectral features, published as four time-series channels.
    spectrum = sdr.measure_spectrum(n_samples=N_SAMPLES)
    for channel, values in spectrum.channel_data.items():
        print(f"{channel} = {values[0]:,.1f}")
finally:
    sdr.close()
