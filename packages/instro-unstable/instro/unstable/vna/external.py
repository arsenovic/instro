## this should be Network.to_json() (ie go in skrf)
def network_to_dict(network) -> dict:
    NETWORK_TO_MEASUREMENT = dict(
        s_time_db="s_time_db",
        s_re="s_re",
        s_im="s_im",
        s_db="s_db",
        s_mag="s_mag",
        s_deg="s_deg",
    )
    FREQUENCY_TO_MEASUREMENT = dict(t_ns="time_ns", f_scaled="f_scaled", f="f_hz")

    frequency = network.frequency
    d1 = {NETWORK_TO_MEASUREMENT[k]: getattr(network, k)[:, 0, 0].tolist() for k in NETWORK_TO_MEASUREMENT}
    d2 = {FREQUENCY_TO_MEASUREMENT[k]: getattr(frequency, k).tolist() for k in FREQUENCY_TO_MEASUREMENT}
    d3 = {"f_unit": frequency.unit}
    data = d1 | d2 | d3
    return data
