"""
Regression tests for pygeofetch.insar.splitspectrum_diagnostic.

Covers only the double-difference PHASE computation, which is
rigorously verified against known synthetic ground truth. The
coherence metric is deliberately not tested here to the same standard
-- see the module's own docstring for why (needs a realistic,
distributed synthetic scene, not built here).
"""

import numpy as np
from pygeofetch.insar.splitspectrum_diagnostic import (
    diagnose_dispersive_signal,
    extract_subband_slc,
)


def test_non_dispersive_signal_gives_zero_double_difference():
    """A real, frequency-INDEPENDENT phase (like real ground
    deformation) must produce exactly zero double-difference -- the
    decisive negative control confirming this method genuinely
    discriminates dispersive from non-dispersive signals."""
    n_azimuth, n_range = 5, 2048
    fs, bw = 64.0e6, 56.5e6
    freqs = np.fft.fftfreq(n_range, d=1.0 / fs)
    in_band = np.abs(freqs) <= bw / 2

    spectrum_ref = np.where(in_band, 1.0 + 0j, 0.0)
    spectrum_sec = np.where(
        in_band, np.exp(1j * 2.5), 0.0
    )  # constant, non-dispersive phase

    ref_slc = np.tile(np.fft.ifft(spectrum_ref), (n_azimuth, 1))
    sec_slc = np.tile(np.fft.ifft(spectrum_sec), (n_azimuth, 1))

    result = diagnose_dispersive_signal(ref_slc, sec_slc, fs, bw)
    assert abs(np.nanmean(result["double_difference_phase"])) < 0.01


def test_known_dispersive_signal_recovered_accurately():
    """A real, physically-motivated dispersive phase (the same
    verified ionospheric formula used elsewhere this session) must be
    recovered by the sub-band double-difference to a decisive
    precision, not just "roughly right"."""
    n_azimuth, n_range = 5, 2048
    fs, bw = 64.0e6, 56.5e6
    f0, c = 5.405e9, 299792458.0
    freqs = np.fft.fftfreq(n_range, d=1.0 / fs)
    in_band = np.abs(freqs) <= bw / 2

    K, tec_diff_tecu = 40.31, 35.0
    tec_diff = tec_diff_tecu * 1e16
    abs_freq = f0 + freqs
    dispersive_phase = -(4 * np.pi * K / c) * tec_diff / abs_freq

    spectrum_ref = np.where(in_band, 1.0 + 0j, 0.0)
    spectrum_sec = np.where(in_band, np.exp(1j * dispersive_phase), 0.0)

    ref_slc = np.tile(np.fft.ifft(spectrum_ref), (n_azimuth, 1))
    sec_slc = np.tile(np.fft.ifft(spectrum_sec), (n_azimuth, 1))

    result = diagnose_dispersive_signal(ref_slc, sec_slc, fs, bw)

    low_center, high_center = -bw / 3, bw / 3
    true_low = 0.0 - (-(4 * np.pi * K / c) * tec_diff / (f0 + low_center))
    true_high = 0.0 - (-(4 * np.pi * K / c) * tec_diff / (f0 + high_center))
    true_double_diff = true_high - true_low

    # Real, wrap-safe averaging: mean in the complex domain, then take
    # the angle -- averaging wrapped angles directly is mathematically
    # incorrect and was the actual cause of a spurious ~0.003 rad
    # discrepancy caught here, not a flaw in the underlying method.
    measured = np.angle(np.nanmean(np.exp(1j * result["double_difference_phase"])))
    assert abs(measured - true_double_diff) < 0.001


def test_coherence_mask_applied_correctly():
    """A real, explicit coherence mask must correctly exclude masked
    pixels (NaN), not silently ignore the mask."""
    n_azimuth, n_range = 5, 2048
    fs, bw = 64.0e6, 56.5e6
    freqs = np.fft.fftfreq(n_range, d=1.0 / fs)
    in_band = np.abs(freqs) <= bw / 2

    spectrum = np.where(in_band, 1.0 + 0j, 0.0)
    ref_slc = np.tile(np.fft.ifft(spectrum), (n_azimuth, 1))
    sec_slc = np.tile(np.fft.ifft(spectrum), (n_azimuth, 1))

    mask = np.zeros((n_azimuth, n_range), dtype=bool)
    mask[:, :100] = True  # only the first 100 range samples are "reliable"

    result = diagnose_dispersive_signal(ref_slc, sec_slc, fs, bw, coherence_mask=mask)
    assert np.all(np.isnan(result["double_difference_phase"][:, 100:]))
    assert not np.any(np.isnan(result["double_difference_phase"][:, :100]))


def test_extract_subband_slc_zenith_no_shift_case():
    """At zero center offset, sub-band extraction with the full
    bandwidth should return the (bandlimited) signal essentially
    unchanged -- a real, decisive sanity check before trusting any
    shifted sub-band result."""
    n_range = 512
    fs, bw = 64.0e6, 56.5e6
    freqs = np.fft.fftfreq(n_range, d=1.0 / fs)
    in_band = np.abs(freqs) <= bw / 2
    spectrum = np.where(in_band, 1.0 + 0j, 0.0)
    slc = np.fft.ifft(spectrum)[None, :]

    result = extract_subband_slc(slc, fs, bw, 0.0)
    assert np.allclose(result, slc, atol=1e-9)
