P6_lightcurve - one night of photometry for 12.410 Problem Set 6
================================================================

A field in Andromeda, observed through the r' filter on the Wallace
Shed C14 (Pier 3) on the night of 2025 Oct 24-25 by last year's class,
with the telescope slightly defocused.  The target, T, is the star
Gaia DR3 356349042800473856, which Gaia DR3 flags as variable.  The
frames have already been reduced and measured: what is here is the
table the photometry pipeline wrote out, one row per 45 s exposure.

    lightcurve_2025-10-25.csv  the photometry, one row per frame
    comparisons.csv            the target (T) and the eight comparison
                               stars (C1-C8): Gaia DR3 source id,
                               position (ra, dec in degrees), Gaia G,
                               BP-RP color and r' magnitude

Columns of lightcurve_2025-10-25.csv:

    date_obs     start of the exposure, UTC
    bjd_tdb      mid-exposure time, BJD_TDB
    exptime      exposure time [s]
    airmass      airmass at mid-exposure
    fwhm         image size [arcsec] (seeing plus the defocus)
    sky          sky level [ADU/pixel]
    flux_T       sky-subtracted flux of the target in a 20-pixel-radius
                 aperture [ADU]
    err_T        its uncertainty [ADU], sqrt(N + n_ap sigma_sky^2)
    flux_C1 ... flux_C8, err_C1 ... err_C8
                 the same for each comparison star

The camera ran at unity gain, so one ADU is one detected photon.
